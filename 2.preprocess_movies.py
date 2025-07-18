import pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer
import numpy as np

# 加载数据
movies = pd.read_csv(r'ml-32m/movies.csv', encoding='utf-8')
tags = pd.read_csv(r'data/tags_processed.csv', encoding='utf-8')

# MovieLens 标准类型
standard_genres = [
    'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary',
    'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
    'Sci-Fi', 'Thriller', 'War', 'Western']

# 关键词映射
genre_keywords = {
    'Action': ['action', 'fight', 'explosion', 'chase'],
    'Adventure': ['adventure', 'journey', 'exploration'],
    'Animation': ['animation', ' Voir'],
    'Comedy': ['comedy', 'funny', 'humor', 'laugh'],
    'Crime': ['crime', 'detective', 'murder'],
    'Documentary': ['documentary', 'real', 'factual'],
    'Drama': ['drama', 'emotional', 'tragedy', 'serious'],
    'Fantasy': ['fantasy', 'magic', 'mythical'],
    'Film-Noir': ['noir', 'dark', 'detective'],
    'Horror': ['horror', 'scary', 'terrifying'],
    'Musical': ['musical', 'song', 'dance'],
    'Mystery': ['mystery', 'detective', 'puzzle'],
    'Romance': ['romance', 'love', 'romantic'],
    'Sci-Fi': ['sci-fi', 'science fiction', 'space', 'future'],
    'Thriller': ['thriller', 'suspense', 'tense'],
    'War': ['war', 'battle', 'military'],
    'Western': ['western', 'cowboy', 'frontier']
}

# 新类型（预定义）
new_genres = ['Psychological', 'Superhero', 'Biographical']

# 加载 Sentence-BERT 模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 量化缺失类型
no_genres_movies = movies[movies['genres'] == '(no genres listed)']
no_genres_count = len(no_genres_movies)
total_movies = len(movies)
print(f"缺失类型的电影数量：{no_genres_count} ({no_genres_count/total_movies*100:.2f}%)")

def infer_genres(movie_id, tags_df, model, standard_genres, new_genres):
    # 获取电影的标签
    movie_tags = tags_df[tags_df['movieId'] == movie_id]['tag'].tolist()
    if not movie_tags:
        return 'Other'

    # 关键词映射
    matched_genres = []
    for tag in movie_tags:
        for genre, keywords in genre_keywords.items():
            if any(keyword in tag.lower() for keyword in keywords):
                if genre not in matched_genres:
                    matched_genres.append(genre)
    
    # 如果关键词映射失败，使用 Sentence-BERT 嵌入
    if not matched_genres:
        tag_text = ' '.join(movie_tags)
        tag_embedding = model.encode(tag_text)
        genre_embeddings = model.encode(standard_genres + new_genres)
        similarities = np.dot(genre_embeddings, tag_embedding) / (
            np.linalg.norm(genre_embeddings, axis=1) * np.linalg.norm(tag_embedding)
        )
        best_genre_idx = np.argmax(similarities)
        matched_genres.append((standard_genres + new_genres)[best_genre_idx])

    return '|'.join(matched_genres) if matched_genres else 'Other'

# 补全缺失类型
movies.loc[movies['genres'] == '(no genres listed)', 'genres'] = movies[movies['genres'] == '(no genres listed)']['movieId'].apply(
    lambda x: infer_genres(x, tags, model, standard_genres, new_genres)
)

# 处理电影类型（转换为二进制特征）
genres = movies['genres'].str.get_dummies('|')
movies = pd.concat([movies, genres], axis=1)

# # 保存预处理后的数据
movies.to_csv(r'data/movies_processed.csv', index=False)

print("类型补全完成，数据已保存至 data/movies_processed.csv！")

# 统计补全后的缺失类型
remaining_no_genres = len(movies[movies['genres'] == 'Other'])
print(f"补全后仍为 'Other' 的电影数量：{remaining_no_genres} ({remaining_no_genres/total_movies*100:.2f}%)")