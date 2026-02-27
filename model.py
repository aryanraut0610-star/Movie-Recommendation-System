import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load data
ratings = pd.read_csv("ml-latest-small/ratings.csv")
movies = pd.read_csv("ml-latest-small/movies.csv")

# Create User-Item Matrix
user_movie_matrix = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)

# Compute similarity between users
user_similarity = cosine_similarity(user_movie_matrix)

similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

def recommend_movies(user_id, n=5):

    # Find similar users
    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:6]
    similar_users_ids = similar_users.index

    movies_watched = user_movie_matrix.loc[user_id]
    recommendations = {}

    for sim_user in similar_users_ids:
        sim_user_ratings = user_movie_matrix.loc[sim_user]

        for movie_id, rating in sim_user_ratings.items():
            if movies_watched[movie_id] == 0 and rating >= 4:
                recommendations[movie_id] = rating

    sorted_movies = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTop {n} Recommendations for User {user_id}:\n")

    count = 0
    for movie_id, rating in sorted_movies:
        title = movies[movies['movieId'] == movie_id]['title'].values[0]
        print(title)
        count += 1
        if count == n:
            break


# Example
recommend_movies(1)