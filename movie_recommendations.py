import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import streamlit as st
from scipy.sparse import load_npz
import joblib
import os
import requests
import random
import gdown

# Google Drive file IDs
MOVIES_CSV_ID = "1ZckRh1dqt81bphsha9OJkpydawjhX7LZ"  
RATINGS_CSV_ID = "1WQ0G96Y_x7McPxrC0aluL9BHBInHSvxS"
COSINE_SIM_ID = "1d3QG1gSr3mcljO_g5LrbRLlT9StM8d2G"
INDICES_ID = "1pDcgxDZvYSNqQulNaU8rWQohYnva7soL"
SVD_MODEL_ID = "1FHMdaOGHrNJIFuk4lQgxPJjhzdd47ZdY"  

# Define paths
DATA_DIR = "data"
PICKLE_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PICKLE_DIR, exist_ok=True)

# Function to download files from Google Drive
def download_file(file_id, dest_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading {file_id} to {dest_path}")
    gdown.download(url, dest_path, quiet=False)

# Test a specific file
download_file(COSINE_SIM_ID, "models/test_cosine_sim.pkl")
print("Download complete.")
print("File exists:", os.path.exists("models/test_cosine_sim.pkl"))

# Download datasets if not already present
MOVIES_CSV_PATH = os.path.join(DATA_DIR, "movies_cleaned.csv")
if not os.path.exists(MOVIES_CSV_PATH):
    download_file(MOVIES_CSV_ID, MOVIES_CSV_PATH)

RATINGS_CSV_PATH = os.path.join(DATA_DIR, "ratings_cleaned.csv")
if not os.path.exists(RATINGS_CSV_PATH):
    download_file(RATINGS_CSV_ID, RATINGS_CSV_PATH)

# Load datasets
@st.cache_data
def load_data():
    movies_df = pd.read_csv(MOVIES_CSV_PATH)
    ratings_df = pd.read_csv(RATINGS_CSV_PATH)
    return movies_df, ratings_df

movies_df, ratings_df = load_data()

# Download and load models
COSINE_SIM_PATH = os.path.join(PICKLE_DIR, "cosine_sim.pkl")
INDICES_PATH = os.path.join(PICKLE_DIR, "indices.pkl")
SVD_MODEL_PATH = os.path.join(PICKLE_DIR, "svd_model.pkl")

if not os.path.exists(COSINE_SIM_PATH):
    download_file(COSINE_SIM_ID, COSINE_SIM_PATH)
if not os.path.exists(INDICES_PATH):
    download_file(INDICES_ID, INDICES_PATH)
if not os.path.exists(SVD_MODEL_PATH):
    download_file(SVD_MODEL_ID, SVD_MODEL_PATH)

@st.cache_resource
def load_models():
    with open(COSINE_SIM_PATH, "rb") as f:
        cosine_sim = pickle.load(f)

    indices = joblib.load(INDICES_PATH)
    indices = {str(k).lower(): v for k, v in indices.items()}  

    svd_model = joblib.load(SVD_MODEL_PATH)  # Load trained SVD model
    
    return cosine_sim, indices, svd_model

cosine_sim, indices, svd_model = load_models()

# TMDb API Key
TMDB_API_KEY = "e73c59ff072f31c2c1ab192f322749e5"

# Function to generate TMDb movie link
def get_tmdb_link(movie_title):
    query = movie_title.replace(" ", "+")
    return f"https://www.themoviedb.org/search?query={query}"

def get_movie_details(movie_title):
    """Fetches the latest poster URL and trailer link from TMDb API."""
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
    
    try:
        response = requests.get(search_url)
        data = response.json()
        
        if "results" in data and len(data["results"]) > 0:
            movie_id = data["results"][0]["id"]
            poster_path = data["results"][0].get("poster_path")
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
            
            # Fetch movie trailer
            trailer_url = None
            video_url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}"
            video_response = requests.get(video_url).json()
            
            if "results" in video_response and len(video_response["results"]) > 0:
                for video in video_response["results"]:
                    if video["site"] == "YouTube" and video["type"] == "Trailer":
                        trailer_url = f"https://www.youtube.com/watch?v={video['key']}"
                        break  # Use the first available trailer

            return poster_url, trailer_url
    except Exception as e:
        print(f"Error fetching details for {movie_title}: {e}")

    return None, None

# Content-Based Recommendation
def content_based_recommend(selected_movies, num_recommendations=10):
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

# Collaborative Filtering Recommendation using SVD
def collaborative_filtering_recommend(selected_ratings, num_recommendations=10):
    predicted_scores = {}

    # Get unique movie IDs (excluding movies already rated by the user)
    movie_ids = set(ratings_df['movieId'].unique()) - set(selected_ratings.keys())

    for movie_id in movie_ids:
        total_score = 0
        weight_sum = 0

        for rated_movie, rating in selected_ratings.items():
            try:
                pred = svd_model.predict(uid=9999, iid=movie_id).est  # Predict rating
                total_score += pred * rating  # Weight by user's given rating
                weight_sum += rating  # Sum of user ratings for normalization
            except:
                continue

        if weight_sum > 0:
            predicted_scores[movie_id] = total_score / weight_sum
        else:
            predicted_scores[movie_id] = 0  # Default low score for unknown movies

    # Sort movies by predicted rating
    sorted_movie_ids = sorted(predicted_scores, key=predicted_scores.get, reverse=True)

    # Introduce randomness by selecting from the top 30 movies
    top_n_candidates = sorted_movie_ids[:30]  # Adjust diversity level
    selected_movies = random.sample(top_n_candidates, min(num_recommendations, len(top_n_candidates)))

    # Return recommended movies
    return movies_df[movies_df['movieId'].isin(selected_movies)][['title']]

# Hybrid Recommendation (Weighted Combination)
def hybrid_recommend(selected_movies, selected_ratings, alpha=0.5, num_recommendations=10):
    content_recs = content_based_recommend(selected_movies, num_recommendations)
    collab_recs = collaborative_filtering_recommend(selected_ratings, num_recommendations)

    combined_scores = {}

    for movie in content_recs:
        combined_scores[movie] = alpha * 1  

    for movie in collab_recs['title'].tolist():
        if movie in combined_scores:
            combined_scores[movie] += (1 - alpha) * 1
        else:
            combined_scores[movie] = (1 - alpha) * 1

    sorted_movies = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
    return sorted_movies[:num_recommendations]

# Apply custom CSS for consistent alignment
st.markdown(
    """
    <style>
        /* Body Background */
        body, .stApp {
            background-color: #050A44 !important; /* Dark Gray (not pure black) */
            color: white !important;
        }
        /* Movie Container (Softer contrast) */
        .movie-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            justify-content: space-between;
            min-height: 420px; 
            width: 220px;
            padding: 12px;
            border-radius: 12px;
            background-color: #1E1E1E; /* Darker gray for card contrast */
            box-shadow: 3px 3px 10px rgba(255, 255, 255, 0.1);
        }
        /* Movie Title */
        .movie-title {
            font-size: 16px;
            font-weight: bold;
            height: 48px;
            line-height: 1.2;
            overflow: hidden;
            text-overflow: ellipsis;
            text-align: center;
            width: 100%;
            padding: 6px 0;
            color: #E50914; /* Netflix Red */
        }
        /* Movie Poster */
        .movie-poster {
            width: 200px;
            height: 300px;
            object-fit: cover;
            border-radius: 10px;
            box-shadow: 2px 2px 8px rgba(255, 255, 255, 0.2);
        }
        /* Buttons */
        .stButton>button {
            background-color: #E50914;
            color: white !important;
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 14px;
            font-weight: bold;
            border: none;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #B20710; /* Darker Red */
        }
        /* Change text color for "Choose a model" */
        .stRadio label {
            color: white !important;
            font-size: 18px !important;
            font-weight: bold !important;
        }
        /* Hyperlinks */
        a {
            color: #fffa65 !important;
            text-decoration: none;
            font-weight: bold;
        }
        a:hover {
            color: #FF3D00 !important; /* Brighter red on hover */
        }
    </style>
    """,
    unsafe_allow_html=True
)

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
                cols = st.columns(3)  # Create 3 columns for horizontal layout

                for i, movie in enumerate(recommendations):
                    poster_url, trailer_url = get_movie_details(movie)

                    with cols[i % 3]:  # Arrange 3 movies per row
                        st.subheader(movie)
                        if poster_url:
                            st.image(poster_url, width=200, caption=movie)  # Display poster
                        st.markdown(f"[More Info]({get_tmdb_link(movie)})")  # TMDb link
                        if trailer_url:
                            st.markdown(f"[Watch Trailer 🎬]({trailer_url})")
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

        if not recommendations.empty:  # Fix: Check if DataFrame is empty
            cols = st.columns(3)  # Arrange in 3 columns per row
            
            for i, movie in enumerate(recommendations['title']):
                poster_url, trailer_url = get_movie_details(movie)

                with cols[i % 3]:  # Distribute across columns
                    st.subheader(movie)
                    if poster_url:
                        st.image(poster_url, width=150, caption=movie)
                    st.markdown(f"[More Info]({get_tmdb_link(movie)})")
                    if trailer_url:
                        st.markdown(f"[Watch Trailer 🎬]({trailer_url})")
        else:
            st.warning("No recommendations found. Try selecting different movies.")

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

    alpha = st.slider("Hybrid weight (0 = Collaborative, 1 = Content-Based)", 0.0, 1.0, 0.5)

    if st.button("Get Recommendations"):
        recommendations = hybrid_recommend(selected_movies, selected_ratings, alpha)

        if recommendations:
            cols = st.columns(3)  # Create 3 columns for horizontal layout

            for i, movie in enumerate(recommendations):
                poster_url, trailer_url = get_movie_details(movie)

                with cols[i % 3]:  # Arrange 3 movies per row
                    st.subheader(movie)
                    if poster_url:
                        st.image(poster_url, width=200, caption=movie)  # Display poster
                    st.markdown(f"[More Info]({get_tmdb_link(movie)})")  # TMDb link
                    if trailer_url:
                        st.markdown(f"[Watch Trailer 🎬]({trailer_url})")
        else:
            st.write("No recommendations found. Try selecting different movies.")
