import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
import joblib

# 1. 生成 ratings_sampled.parquet
ratings = pd.read_csv('data/ratings_processed.csv')
ratings_sampled = ratings.sample(frac=0.1, random_state=42)
ratings_sampled.to_parquet('data/ratings_sampled.parquet', index=False)

# 2. 训练并保存 svd_model.joblib
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings_sampled[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
svd = SVD(n_factors=100, n_epochs=10, random_state=42)
svd.fit(trainset)
joblib.dump(svd, 'data/svd_model.joblib')

# 3. 生成 content_sim.npy
movies = pd.read_csv('data/movies_processed.csv')
movie_counts = ratings['movieId'].value_counts()
top_movies = movie_counts.head(10000).index
movies = movies[movies['movieId'].isin(top_movies)]
genre_columns = [col for col in movies.columns if col not in ['movieId', 'title', 'genres']]
movie_features = movies.set_index('movieId')[genre_columns].fillna(0)
content_sim = cosine_similarity(movie_features)
np.save('data/content_sim.npy', content_sim)