import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import streamlit as st
from scipy.sparse import load_npz
import joblib
import os

# Define file paths
CLEANED_DATA_PATH = r"C:\Users\wongj\Documents\RDS Y2S2\Artificial Intelligence\MovieRecommendationSystem\CleanedData"
PICKLE_PATH = r"C:\Users\wongj\Documents\RDS Y2S2\Artificial Intelligence\MovieRecommendationSystem\Pickle"

@st.cache_data
def load_data():
    movies_df = pd.read_csv(f"{CLEANED_DATA_PATH}/movies_cleaned.csv")
    ratings_df = pd.read_csv(f"{CLEANED_DATA_PATH}/ratings_cleaned.csv")
    return movies_df, ratings_df

movies_df, ratings_df = load_data()

# Load precomputed similarity matrices
@st.cache_resource
def load_pickle_files():
    with open(f"{PICKLE_PATH}/cosine_sim.pkl", "rb") as f:
        cosine_sim = pickle.load(f)  # Load as a dense matrix

    indices = joblib.load(f"{PICKLE_PATH}/indices.pkl")  # Load index mapping
    indices = {str(k).lower(): v for k, v in indices.items()}  # Normalize keys to lowercase

    return cosine_sim, indices

cosine_sim, indices = load_pickle_files()

# TMDb API Key
TMDB_API_KEY = "e73c59ff072f31c2c1ab192f322749e5"

# Function to generate TMDb movie link
def get_tmdb_link(movie_title):
    query = movie_title.replace(" ", "+")
    return f"https://www.themoviedb.org/search?query={query}"

# Content-Based Recommendation
def content_based_recommend(selected_movies, num_recommendations=5):
    recommended_movies = []
    for title in selected_movies:
        title = title.strip().lower()
        if title not in indices:
            continue
        idx = indices[title]
        sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
        movie_indices = [i[0] for i in sim_scores]
        recommended_movies.extend(movies_df.iloc[movie_indices]['title'].tolist())
    return list(set(recommended_movies))[:num_recommendations]

# Collaborative Filtering Recommendation
def collaborative_filtering_recommend(selected_ratings, num_recommendations=5):
    user_item_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    item_similarity = cosine_similarity(user_item_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)
    
    scores = np.zeros(len(user_item_matrix.columns))
    for movie_id, rating in selected_ratings.items():
        if movie_id in item_similarity_df:
            scores += item_similarity_df[movie_id] * rating
    
    sorted_movie_ids = [movie_id for movie_id in np.argsort(scores)[::-1] if movie_id not in selected_ratings][:num_recommendations]
    return movies_df[movies_df['movieId'].isin(sorted_movie_ids)][['title']]

# Hybrid Recommendation
def hybrid_recommend(selected_movies, selected_ratings, num_recommendations=5):
    recommended_movies = set()
    
    # Content-Based Filtering
    for title in selected_movies:
        title = title.strip().lower()
        if title not in indices:
            continue
        idx = indices[title]
        sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
        movie_indices = [i[0] for i in sim_scores]
        recommended_movies.update(movies_df.iloc[movie_indices]['title'].tolist())
    
    # Collaborative Filtering
    user_item_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    item_similarity = cosine_similarity(user_item_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)
    
    scores = np.zeros(len(user_item_matrix.columns))
    for movie_id, rating in selected_ratings.items():
        if movie_id in item_similarity_df:
            scores += item_similarity_df[movie_id] * rating
    
    sorted_movie_ids = [movie_id for movie_id in np.argsort(scores)[::-1] if movie_id not in selected_ratings][:num_recommendations]
    recommended_movies.update(movies_df[movies_df['movieId'].isin(sorted_movie_ids)]['title'].tolist())
    
    return list(recommended_movies)[:num_recommendations]

# Streamlit UI
st.title("Movie Recommendation System")
st.write("""Choose a recommendation model to get personalized movie recommendations.""")

model_option = st.radio("Choose a model", ["Content-Based", "Collaborative Filtering", "Hybrid"])

if model_option == "Content-Based":
    selected_movies = st.multiselect("Select movies you like:", movies_df['title'].tolist())
    if st.button("Get Recommendations"):
        if selected_movies:
            recommendations = content_based_recommend(selected_movies)
            if recommendations:
                for movie in recommendations:
                    st.subheader(movie)
                    st.markdown(f"[More Info]({get_tmdb_link(movie)})")
            else:
                st.write("No recommendations found. Try selecting different movies.")
        else:
            st.warning("Please select at least one movie.")

elif model_option == "Collaborative Filtering":
    st.write("Select up to 3 movies and rate them:")
    selected_ratings = {}
    for i in range(3):
        movie = st.selectbox(f"Movie {i+1}", movies_df['title'].tolist(), index=i)
        rating = st.slider(f"Rate {movie}", 1, 5, 3)
        selected_ratings[movies_df[movies_df['title'] == movie]['movieId'].values[0]] = rating
    if st.button("Get Recommendations"):
        recommendations = collaborative_filtering_recommend(selected_ratings)
        for movie in recommendations['title']:
            st.subheader(movie)
            st.markdown(f"[More Info]({get_tmdb_link(movie)})")

elif model_option == "Hybrid":
    # Content-Based Selection
    selected_movies = st.multiselect("Select movies you like:", movies_df['title'].tolist())

    # Collaborative Filtering Ratings
    st.write("Rate up to 3 movies:")
    selected_ratings = {}
    for i in range(3):
        movie = st.selectbox(f"Movie {i+1}", movies_df['title'].tolist(), index=i)
        rating = st.slider(f"Rate {movie}", 1, 5, 3)
        selected_ratings[movies_df[movies_df['title'] == movie]['movieId'].values[0]] = rating

    if st.button("Get Recommendations"):
        recommendations = set()

        if selected_movies:
            content_recs = content_based_recommend(selected_movies)
            recommendations.update(content_recs)

        if selected_ratings:
            collab_recs = collaborative_filtering_recommend(selected_ratings)
            recommendations.update(collab_recs['title'].tolist())

        if recommendations:
            for movie in recommendations:
                st.subheader(movie)
                st.markdown(f"[More Info]({get_tmdb_link(movie)})")
        else:
            st.write("No recommendations found. Try selecting different movies.")

