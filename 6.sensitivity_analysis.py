import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time

# 设置 Matplotlib 样式，使用 seaborn 调色板，字体为 Times New Roman，字体大小 18
plt.style.use('seaborn')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

# 加载数据
movies = pd.read_csv('data/movies_processed.csv', encoding='utf-8')
ratings = pd.read_csv('data/ratings_processed.csv', encoding='utf-8')
links = pd.read_csv('ml-32m/links.csv', encoding='utf-8')

# 初始采样和限制电影
ratings = ratings.sample(frac=0.1, random_state=42)
movie_counts = ratings['movieId'].value_counts()
top_movies = movie_counts.head(10000).index
movies = movies[movies['movieId'].isin(top_movies)]
ratings = ratings[ratings['movieId'].isin(top_movies)]
links = links[links['movieId'].isin(top_movies)]

# 训练 SVD 模型
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
svd = SVD(random_state=42, n_epochs=10)
print("Training SVD Model...")
with tqdm(total=1, desc="SVD Training") as pbar:
    svd.fit(trainset)
    pbar.update(1)
svd_predictions = svd.test(testset)
svd_rmse = accuracy.rmse(svd_predictions, verbose=False)
svd_mae = accuracy.mae(svd_predictions, verbose=False)

# 加载预计算的相似性矩阵
content_sim = np.load('data/content_sim.npy')
genre_columns = [col for col in movies.columns if col not in ['movieId', 'title', 'genres']]
movie_features = movies.set_index('movieId')[genre_columns].fillna(0)
movie_ids = movies['movieId'].values

# 推荐函数
def hybrid_recommend(user_id, movie_id=None, genres=None, svd_model=svd, sim_matrix=content_sim, movie_ids=movie_ids, top_k=10, weight=0.5):
    with tqdm(total=1, desc="Hybrid Recommendation") as pbar:
        if movie_id or genres:
            similar_movies = content_based_recommend(movie_id, genres, sim_matrix, movie_features, movie_ids, top_k * 2)
        else:
            similar_movies = content_based_recommend(top_k=top_k * 2)
        hybrid_scores = []
        for sim_movie_id in similar_movies:
            svd_pred = svd_model.predict(user_id, sim_movie_id).est
            sim_score = sim_matrix[np.where(movie_ids == movie_id)[0][0]][np.where(movie_ids == sim_movie_id)[0][0]] if movie_id else 1.0
            hybrid_score = weight * svd_pred + (1 - weight) * sim_score
            hybrid_scores.append((sim_movie_id, hybrid_score))
        hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
        pbar.update(1)
    return [movie_id for movie_id, _ in hybrid_scores[:top_k]]

def content_based_recommend(movie_id=None, genres=None, sim_matrix=content_sim, movie_features=movie_features, movie_ids=movie_ids, top_k=10):
    if movie_id:
        if movie_id not in movie_ids:
            return []
        idx = np.where(movie_ids == movie_id)[0][0]
        sim_scores = list(enumerate(sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        return [movie_ids[i] for i, _ in sim_scores[1:top_k+1]]
    elif genres:
        genre_vector = np.zeros(len(genre_columns))
        for genre in genres:
            if genre in genre_columns:
                genre_vector[genre_columns.index(genre)] = 1
        sim_scores = cosine_similarity([genre_vector], movie_features)[0]
        sim_scores = list(enumerate(sim_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        return [movie_ids[i] for i, _ in sim_scores[:top_k]]
    else:
        avg_ratings = ratings.groupby('movieId')['rating'].mean().sort_values(ascending=False)
        return avg_ratings[avg_ratings.index.isin(movie_ids)].head(top_k).index.tolist()

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    user_est_true = {}
    for pred in tqdm(predictions, desc="Evaluating Precision@10 and Recall@10"):
        user = pred.uid
        if user not in user_est_true:
            user_est_true[user] = []
        user_est_true[user].append((pred.est, pred.r_ui))
    
    precisions = []
    recalls = []
    for user, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for _, true_r in user_ratings)
        n_rec_k = sum((est >= threshold) for est, _ in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for est, true_r in user_ratings[:k])
        precisions.append(n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0)
        recalls.append(n_rel_and_rec_k / n_rel if n_rel != 0 else 0)
    
    return np.mean(precisions), np.mean(recalls)

# 灵敏度分析
sensitivity_results = {'Sampling Ratio': [], 'RMSE': [], 'MAE': [], 'Precision@10': [], 'Recall@10': []}
sampling_ratios = [0.05, 0.1, 0.15, 0.2, 0.3]
for ratio in sampling_ratios:
    sampled_ratings = ratings.sample(frac=ratio, random_state=42)
    data = Dataset.load_from_df(sampled_ratings[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    svd = SVD(random_state=42, n_epochs=10)
    svd.fit(trainset)
    predictions = svd.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    precision, recall = precision_recall_at_k(predictions)
    sensitivity_results['Sampling Ratio'].append(ratio)
    sensitivity_results['RMSE'].append(rmse)
    sensitivity_results['MAE'].append(mae)
    sensitivity_results['Precision@10'].append(precision)
    sensitivity_results['Recall@10'].append(recall)

# 数据规模灵敏度分析
scale_results = {'Movie Count': [], 'Computation Time (s)': [], 'Diversity (%)': [], 'RMSE': []}
movie_counts = [5000, 7500, 10000, 12500, 15000]
for count in movie_counts:
    start_time = time.time()
    current_movie_counts = ratings['movieId'].value_counts()
    top_movies = current_movie_counts.head(count).index
    movies_subset = movies[movies['movieId'].isin(top_movies)]
    movie_features = movies_subset.set_index('movieId')[genre_columns].fillna(0)
    content_sim = cosine_similarity(movie_features)
    end_time = time.time()
    computation_time = end_time - start_time
    diversity = len(set(movies_subset['genres'].str.split('|').explode())) / len(genre_columns) * 100
    # 训练 SVD 模型以获取 RMSE
    sampled_ratings = ratings[ratings['movieId'].isin(top_movies)].sample(frac=0.1, random_state=42)
    data = Dataset.load_from_df(sampled_ratings[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    svd = SVD(random_state=42, n_epochs=10)
    svd.fit(trainset)
    predictions = svd.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    scale_results['Movie Count'].append(count)
    scale_results['Computation Time (s)'].append(computation_time)
    scale_results['Diversity (%)'].append(diversity)
    scale_results['RMSE'].append(rmse)

# 性能评估
web_response_times = []
for _ in range(10):  # 模拟 10 次请求
    start_time = time.time()
    sample_user = ratings['userId'].sample(1).iloc[0]
    hybrid_recs = hybrid_recommend(sample_user)
    end_time = time.time()
    web_response_times.append(end_time - start_time)

# 可视化灵敏度分析
plt.figure(figsize=(10, 6))
plt.plot(sensitivity_results['Sampling Ratio'], sensitivity_results['RMSE'], marker='o', linewidth=2, label='RMSE', color='#1f77b4')
plt.plot(sensitivity_results['Sampling Ratio'], sensitivity_results['MAE'], marker='s', linewidth=2, label='MAE', color='#ff7f0e')
plt.plot(sensitivity_results['Sampling Ratio'], sensitivity_results['Precision@10'], marker='^', linewidth=2, label='Precision@10', color='#2ca02c')
plt.xlabel('Sampling Ratio')
plt.ylabel('Metric Value')
plt.title('Sensitivity Analysis: Sampling Ratio Impact')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('reports/sampling_sensitivity.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(scale_results['Movie Count'], scale_results['Computation Time (s)'], marker='o', linewidth=2, label='Computation Time (s)', color='#1f77b4')
plt.plot(scale_results['Movie Count'], scale_results['Diversity (%)'], marker='s', linewidth=2, label='Diversity (%)', color='#ff7f0e')
plt.plot(scale_results['Movie Count'], scale_results['RMSE'], marker='^', linewidth=2, label='RMSE', color='#2ca02c')
plt.xlabel('Movie Count')
plt.ylabel('Value')
plt.title('Sensitivity Analysis: Movie Count Impact')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('reports/scale_sensitivity.png')
plt.show()

# 可视化性能评估
plt.figure(figsize=(10, 6))
plt.boxplot(web_response_times, patch_artist=True, boxprops=dict(facecolor='#1f77b4', color='black'),
            medianprops=dict(color='black'), whiskerprops=dict(color='black'), capprops=dict(color='black'))
plt.xlabel('Web Request')
plt.ylabel('Response Time (s)')
plt.title('Web Response Time Distribution')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('reports/web_response_time.png')
plt.show()

# 添加混杂推荐与单一推荐对比
svd_recs = [(mid, svd.predict(ratings['userId'].sample(1).iloc[0], mid).est) for mid in movie_ids]
svd_recs = sorted(svd_recs, key=lambda x: x[1], reverse=True)[:10]
content_recs = content_based_recommend(movie_id=ratings['movieId'].sample(1).iloc[0])
hybrid_recs = hybrid_recommend(ratings['userId'].sample(1).iloc[0])
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), [svd.predict(ratings['userId'].sample(1).iloc[0], mid).est for mid, _ in svd_recs[:10]], marker='o', linewidth=2, label='SVD', color='#1f77b4')
plt.plot(range(1, 11), [4.0] * 10, marker='s', linewidth=2, label='Content-Based', color='#ff7f0e', linestyle='--')  # 简化内容推荐得分
plt.plot(range(1, 11), [svd.predict(ratings['userId'].sample(1).iloc[0], mid).est for mid in hybrid_recs], marker='^', linewidth=2, label='Hybrid', color='#2ca02c')
plt.xlabel('Recommendation Rank')
plt.ylabel('Predicted Score')
plt.title('Comparison of Recommendation Scores')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('reports/recommendation_comparison.png')
plt.show()

# 添加用户评分分布
plt.figure(figsize=(10, 6))
ratings['rating'].hist(bins=20, color='#1f77b4', edgecolor='black')
plt.xlabel('Rating Value')
plt.ylabel('Frequency')
plt.title('User Rating Distribution')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('reports/rating_distribution.png')
plt.show()

print(f"Sensitivity Results - Sampling: RMSE {sensitivity_results['RMSE']}, MAE {sensitivity_results['MAE']}")
print(f"Sensitivity Results - Scale: Time {scale_results['Computation Time (s)']}, Diversity {scale_results['Diversity (%)']}, RMSE {scale_results['RMSE']}")
print(f"Performance - Web Response Time: {np.mean(web_response_times):.2f}s")