import streamlit as st
import pandas as pd
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

# Compute similarity
user_similarity = cosine_similarity(user_movie_matrix)

similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

def recommend_movies(user_id, n=5):

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

    recommended_titles = []

    for movie_id, rating in sorted_movies[:n]:
        title = movies[movies['movieId'] == movie_id]['title'].values[0]
        recommended_titles.append(title)

    return recommended_titles


st.title("🎬 Movie Recommendation System")

user_id = st.number_input(
    "Enter User ID",
    min_value=int(user_movie_matrix.index.min()),
    max_value=int(user_movie_matrix.index.max()),
    step=1
)

if st.button("Get Recommendations"):
    results = recommend_movies(user_id)

    if results:
        st.subheader("Recommended Movies:")
        for movie in results:
            st.write(movie)
    else:
        st.write("No recommendations found.")