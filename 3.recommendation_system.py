import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
movies = pd.read_csv('data/movies_processed.csv', encoding='utf-8')
ratings = pd.read_csv('data/ratings_processed.csv', encoding='utf-8')

# 采样数据（10%）以加速训练
ratings = ratings.sample(frac=0.3, random_state=42)
print(f"采样后的评分数据量：{len(ratings)}")

# 限制电影数量（热门电影：评分次数前 10,000）
movie_counts = ratings['movieId'].value_counts()
top_movies = movie_counts.head(10000).index
movies = movies[movies['movieId'].isin(top_movies)]
ratings = ratings[ratings['movieId'].isin(top_movies)]
print(f"限制后的电影数量：{len(movies)}")

#----------------------------------------------------------- 1. SVD 模型 --------------------------------------------------------------------
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

print("训练 SVD 模型...")
svd = SVD(random_state=42, n_epochs=10)  # 减少迭代次数
with tqdm(total=1, desc="SVD 训练") as pbar:
    svd.fit(trainset)
    pbar.update(1)

svd_predictions = svd.test(testset)
svd_rmse = accuracy.rmse(svd_predictions, verbose=False)
svd_mae = accuracy.mae(svd_predictions, verbose=False)
print(f"SVD - RMSE: {svd_rmse:.4f}, MAE: {svd_mae:.4f}")

#---------------------------------------------------------- 2. 基于内容的推荐 -----------------------------------------------------------------------
genre_columns = [col for col in movies.columns if col not in ['movieId', 'title', 'genres']]
movie_features = movies.set_index('movieId')[genre_columns].fillna(0)
movie_ids = movies['movieId'].values  # 用于索引映射

# 分批计算相似性
def compute_similarity_in_batches(features, batch_size=1000):
    n_movies = features.shape[0]
    sim_matrix = np.zeros((n_movies, n_movies))
    for i in tqdm(range(0, n_movies, batch_size), desc="计算相似性"):
        start_i = i
        end_i = min(i + batch_size, n_movies)
        batch_sim = cosine_similarity(features.iloc[start_i:end_i], features)
        sim_matrix[start_i:end_i] = batch_sim
    return sim_matrix

content_sim = compute_similarity_in_batches(movie_features)

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

#------------------------------------------------------------- 3. 混合推荐 -----------------------------------------------------------------------
def hybrid_recommend(user_id, movie_id=None, genres=None, svd_model=svd, sim_matrix=content_sim, movie_ids=movie_ids, top_k=10):
    with tqdm(total=1, desc="混合推荐") as pbar:
        if movie_id or genres:
            similar_movies = content_based_recommend(movie_id, genres, sim_matrix, movie_features, movie_ids, top_k * 2)
        else:
            similar_movies = content_based_recommend(top_k=top_k * 2)
        hybrid_scores = []
        for sim_movie_id in similar_movies:
            svd_pred = svd_model.predict(user_id, sim_movie_id).est
            sim_score = sim_matrix[np.where(movie_ids == movie_id)[0][0]][np.where(movie_ids == sim_movie_id)[0][0]] if movie_id else 1.0
            hybrid_score = 0.5 * svd_pred + 0.5 * sim_score
            hybrid_scores.append((sim_movie_id, hybrid_score))
        hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
        pbar.update(1)
    return [movie_id for movie_id, _ in hybrid_scores[:top_k]]

#------------------------------------------------------ 4. 评估 Precision@10 和 Recall@10 ------------------------------------------------------
def precision_recall_at_k(predictions, k=10, threshold=3.5):
    user_est_true = {}
    for pred in tqdm(predictions, desc="评估 Precision@10 和 Recall@10"):
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

svd_precision, svd_recall = precision_recall_at_k(svd_predictions, k=10)
print(f"SVD - Precision@10: {svd_precision:.4f}, Recall@10: {svd_recall:.4f}")

#--------------------------------------------------- 测试推荐 ----------------------------------------------------------------------
sample_user = ratings['userId'].sample(1).iloc[0]
sample_movie = ratings['movieId'].sample(1).iloc[0] 
with tqdm(total=len(movies['movieId']), desc="SVD 推荐") as pbar:
    svd_recs = [(mid, svd.predict(sample_user, mid).est) for mid in movies['movieId']]
    pbar.update(len(movies['movieId']))
svd_recs = sorted(svd_recs, key=lambda x: x[1], reverse=True)[:10]
svd_recs = [mid for mid, _ in svd_recs]
content_recs = content_based_recommend(movie_id=sample_movie)
content_genre_recs = content_based_recommend(genres=['Comedy', 'Animation'])
random_recs = content_based_recommend()
hybrid_recs = hybrid_recommend(sample_user, movie_id=sample_movie)

print(f"SVD 推荐（用户 {sample_user}）：{[movies[movies['movieId'] == mid]['title'].iloc[0] for mid in svd_recs]}")
print(f"基于内容（电影 {sample_movie}）：{[movies[movies['movieId'] == mid]['title'].iloc[0] for mid in content_recs]}")
print(f"基于类型（Comedy, Animation）：{[movies[movies['movieId'] == mid]['title'].iloc[0] for mid in content_genre_recs]}")
print(f"随机推荐（热门）：{[movies[movies['movieId'] == mid]['title'].iloc[0] for mid in random_recs]}")
print(f"混合推荐（用户 {sample_user}, 电影 {sample_movie}）：{[movies[movies['movieId'] == mid]['title'].iloc[0] for mid in hybrid_recs]}")

#------------------------------------------------------- 可视化性能 -------------------------------------------------------------------------
metrics = {'SVD': [svd_rmse, svd_mae, svd_precision, svd_recall]}
metrics_df = pd.DataFrame(metrics, index=['RMSE', 'MAE', 'Precision@10', 'Recall@10'])

ax = metrics_df.plot(kind='bar', figsize=(12, 8), legend=False)

# 设置标题和标签字体大小
# ax.set_title('算法性能比较', fontsize=16)
ax.set_ylabel('value', fontsize=32)
ax.set_xticklabels(metrics_df.index, rotation=0, fontsize=32)  # 不旋转横坐标标签
ax.tick_params(axis='y', labelsize=32)

# 保存和展示图像
plt.tight_layout()
plt.savefig('reports/algorithm_performance.png')
plt.show()