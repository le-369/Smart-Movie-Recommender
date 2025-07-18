import pandas as pd
import numpy as np
from surprise import SVD
from surprise import Dataset, Reader
from flask import Flask, render_template, request, session, jsonify
from tqdm import tqdm
import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
import threading
import time
import random
import re
import logging

app = Flask(__name__)
app.secret_key = os.urandom(16)

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

log_queue = []

class Recommender:
    def __init__(self):
        self.movies = pd.read_csv('data/movies_processed.csv')
        self.ratings = pd.read_parquet('data/ratings_sampled.parquet')
        self.ratings['rating'] = self.ratings['rating'].astype(float)
        self.links = pd.read_csv('ml-32m/links.csv').dropna(subset=['tmdbId'])
        self.movies = self.movies.merge(self.links[['movieId', 'tmdbId']], on='movieId', how='left')
        with open('movies_tmdb.json', 'r', encoding='utf-8') as f:
            raw_tmdb_data = json.load(f)
            self.tmdb_movies = {str(int(entry['tmdb_id'].split('-')[0])): entry for entry in raw_tmdb_data if '-' in entry['tmdb_id']}
        self.svd = load('data/svd_model.joblib')  # 保留但不使用
        self.movie_features = self.movies.set_index('movieId')[['Action', 'Adventure', 'Animation', 'Biographical', 'Children', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Other', 'Psychological', 'Romance', 'Sci-Fi', 'Superhero', 'Thriller', 'War', 'Western']].fillna(0)
        self.movie_ids = self.movies['movieId'].values
        self.content_sim = np.load('data/content_sim.npy') if os.path.exists('data/content_sim.npy') else cosine_similarity(self.movie_features)
        if not os.path.exists('data/content_sim.npy'):
            np.save('data/content_sim.npy', self.content_sim)
        logger.debug(f"Loaded {len(self.movies)} movies, {len(self.tmdb_movies)} TMDB entries with numeric keys")

recommender = Recommender()

def get_movie_details(movie_id):
    movie = recommender.movies[recommender.movies['movieId'] == movie_id]
    if movie.empty:
        logger.error(f"Movie ID {movie_id} not found in movies dataset")
        return {'error': '电影未找到'}
    tmdb_id = str(int(movie['tmdbId'].iloc[0])) if pd.notna(movie['tmdbId'].iloc[0]) else None
    logger.debug(f"Processing movie_id {movie_id}, tmdb_id: {tmdb_id}")
    if tmdb_id and tmdb_id in recommender.tmdb_movies:
        tmdb_movie = recommender.tmdb_movies[tmdb_id]
        if all(key in tmdb_movie for key in ['title', 'description', 'poster_url', 'director']):
            avg_rating = recommender.ratings[recommender.ratings['movieId'] == movie_id]['rating'].mean() if movie_id in recommender.ratings['movieId'].values else None
            rating = round(avg_rating, 1) if avg_rating is not None else "未知"
            return {
                'title': tmdb_movie['title'],
                'description': tmdb_movie['description'],
                'poster_url': tmdb_movie['poster_url'],
                'vote_average': rating,
                'director': tmdb_movie['director'],
                'movieId': int(movie_id),
                'tmdbId': tmdb_id
            }
    logger.warning(f"TMDB data missing for movie_id {movie_id}, tmdb_id {tmdb_id}")
    # 仅返回错误，避免展示无 TMDB 数据电影
    return {'error': 'TMDB 数据缺失'}

def content_based_recommend(description, top_k=16):
    genres, _ = parse_description(description)
    if not genres:
        movie_ids = list(recommender.movie_ids)
        random.shuffle(movie_ids)
    else:
        genre_vector = np.zeros(len(recommender.movie_features.columns))
        for genre in genres:
            if genre in recommender.movie_features.columns:
                genre_vector[recommender.movie_features.columns.get_loc(genre)] = 1
        sim_scores = cosine_similarity([genre_vector], recommender.movie_features)[0]
        sim_scores = list(enumerate(sim_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        movie_ids = [recommender.movie_ids[i] for i, _ in sim_scores]
    recommendations = []
    candidate_ids = movie_ids[:top_k * 4]  # 增加候选集以确保足够匹配
    random.shuffle(candidate_ids)
    for mid in candidate_ids:
        details = get_movie_details(mid)
        if details and 'error' not in details:
            recommendations.append(details)
        if len(recommendations) >= top_k:
            break
    return recommendations

def hybrid_recommend(user_id, description, top_k=16):
    genres, _ = parse_description(description)
    similar_movies = content_based_recommend(description, top_k * 4)
    hybrid_scores = []
    for movie in similar_movies:
        mid = movie['movieId']
        svd_pred = recommender.svd.predict(user_id, mid).est
        genre_vector = np.zeros(len(recommender.movie_features.columns))
        for genre in genres:
            if genre in recommender.movie_features.columns:
                genre_vector[recommender.movie_features.columns.get_loc(genre)] = 1
        sim_score = cosine_similarity([genre_vector], [recommender.movie_features.loc[mid]])[0][0] if genres else 0.5
        hybrid_score = 0.5 * svd_pred + 0.5 * sim_score
        hybrid_scores.append((mid, hybrid_score))
    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
    recommendations = []
    for mid, _ in hybrid_scores:
        details = get_movie_details(mid)
        if details and 'error' not in details:
            recommendations.append(details)
        if len(recommendations) >= top_k:
            break
    return recommendations

def parse_description(description):
    genres = []
    keywords = []
    description = description.lower()
    genre_map = {
        '科幻': 'Sci-Fi', '动作': 'Action', '冒险': 'Adventure', '动画': 'Animation',
        '传记': 'Biographical', '儿童': 'Children', '儿童向': "Children's", '喜剧': 'Comedy',
        '犯罪': 'Crime', '纪录片': 'Documentary', '剧情': 'Drama', '奇幻': 'Fantasy',
        '黑色电影': 'Film-Noir', '恐怖': 'Horror', 'IMAX': 'IMAX', '音乐': 'Musical',
        '悬疑': 'Mystery', '其他': 'Other', '心理': 'Psychological', '爱情': 'Romance',
        '超级英雄': 'Superhero', '惊悚': 'Thriller', '战争': 'War', '西部': 'Western'
    }
    for key, mapped_genre in genre_map.items():
        if key in description or mapped_genre.lower() in description:
            genres.append(mapped_genre)
    keywords.extend(re.findall(r'\b\w+\b', description))
    return genres, keywords

def search_movies(query):
    query = query.lower()
    matched_movies = recommender.movies[recommender.movies['title'].str.lower().str.contains(query, na=False)]
    recommendations = []
    for _, row in matched_movies.head(10).iterrows():
        details = get_movie_details(row['movieId'])
        if details and 'error' not in details:
            recommendations.append(details)
    return recommendations

def get_trending_movies(top_k=8):
    trending = recommender.ratings.groupby('movieId')['rating'].mean().sort_values(ascending=False).head(top_k * 4)
    trending_movies = []
    seen_ids = set()
    for mid in trending.index:
        if mid not in seen_ids:
            details = get_movie_details(mid)
            if details and 'error' not in details:
                trending_movies.append({
                    'title': details['title'],
                    'movieId': int(mid),
                    'poster_url': details['poster_url'],
                    'vote_average': details['vote_average']
                })
                seen_ids.add(mid)
            if len(trending_movies) >= top_k:
                break
    if len(trending_movies) < top_k:
        additional_ids = [mid for mid in recommender.movie_ids if mid not in seen_ids]
        random.shuffle(additional_ids)
        for mid in additional_ids:
            if len(trending_movies) >= top_k:
                break
            details = get_movie_details(mid)
            if details and 'error' not in details:
                trending_movies.append({
                    'title': details['title'],
                    'movieId': int(mid),
                    'poster_url': details['poster_url'],
                    'vote_average': details['vote_average']
                })
    return trending_movies

def log_recommendation(data):
    log_queue.append(data)
    if len(log_queue) > 100:
        with open('reports/recommendations.txt', 'a', encoding='utf-8') as f:
            f.write('\n'.join(log_queue))
            log_queue.clear()

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    error = None
    trending = get_trending_movies()
    history = session.get('history', [])
    user = session.get('user', None)

    if request.method == 'POST':
        user_id = request.form.get('user_id', session.get('temp_user_id', np.random.randint(1000000, 9999999)))
        session['temp_user_id'] = user_id if not user else user.get('id')
        description = request.form.get('description', '').strip()
        recommendation_type = request.form.get('recommendation_type', 'description')
        search_query = request.form.get('search_query')
        refresh_trending = request.form.get('refresh_trending')

        if refresh_trending:
            trending = get_trending_movies()
        elif search_query:
            matched_movies = search_movies(search_query)
            recommendations = matched_movies
            if not recommendations:
                error = "未找到匹配的电影"
        else:
            try:
                user_id = int(user_id)
                if recommendation_type == 'description' and description:
                    recommendations = content_based_recommend(description, top_k=16)
                elif recommendation_type == 'hybrid' and description:
                    if user_id in recommender.ratings['userId'].values or 'temp_user_id' in session:
                        recommendations = hybrid_recommend(user_id, description, top_k=16)
                    else:
                        error = "用户 ID 不存在"
                else:
                    error = "请选择有效的推荐类型或输入描述"

                if not error and not recommendations:
                    error = "未找到有效推荐结果"
                elif recommendations:
                    history.append({'type': recommendation_type, 'results': recommendations, 'time': time.strftime('%Y-%m-%d %H:%M')})
                    session['history'] = history[-5:]
                    log_data = f"\n时间: {time.strftime('%Y-%m-%d %H:%M')}\n推荐类型: {recommendation_type}\n用户 ID: {user_id}\n描述: {description}\n推荐结果: {[r['title'] for r in recommendations]}\n"
                    log_recommendation(log_data)
            except Exception as e:
                logger.error(f"Recommendation error: {str(e)}")
                error = f"错误: {str(e)}"

    return render_template('index.html', genres=recommender.movie_features.columns.tolist(), recommendations=recommendations, error=error, trending=trending, history=history, user=user)

@app.route('/get_movie_info', methods=['POST'])
def get_movie_info():
    try:
        movie_id = int(request.form.get('movie_id'))
        details = get_movie_details(movie_id)
        if 'error' in details:
            logger.error(f"Failed to get details for movie_id {movie_id}: {details['error']}")
            return jsonify({'error': details['error']})
        logger.debug(f"Successfully retrieved details for movie_id {movie_id}: {details}")
        return jsonify(details)
    except Exception as e:
        logger.error(f"Exception in get_movie_info for movie_id {movie_id}: {str(e)}")
        return jsonify({'error': '电影信息加载失败，请稍后重试'})

@app.route('/login', methods=['POST'])
def login():
    try:
        username = request.form.get('username')
        password = request.form.get('password')
        if username and password:
            session['user'] = {'id': np.random.randint(1000000, 9999999), 'username': username}
            return jsonify({'status': 'success'})
        return jsonify({'error': '登录失败'})
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({'error': '登录失败'})

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user', None)
    session.pop('temp_user_id', None)
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    if not os.path.exists('reports'):
        os.makedirs('reports')
    def flush_log_queue():
        if log_queue:
            with open('reports/recommendations.txt', 'a', encoding='utf-8') as f:
                f.write('\n'.join(log_queue))
                log_queue.clear()
    threading.Thread(target=flush_log_queue, daemon=True).start()
    app.run(debug=True)