import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
import dagshub
import joblib
import os

# Set up MLflow and DagsHub configuration using environment variables
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/mariouskono/modelll.mlflow'
os.environ['DAGSHUB_TOKEN'] = os.getenv("DAGSHUB_TOKEN")  # Secure via GitHub Secrets

# Initialize DagsHub with token from env
dagshub.init(repo_owner='mariouskono', repo_name='modelll', mlflow=True)

try:
    # Load the dataset
    df = pd.read_csv('dataset_tempat_wisata_bali_processed.csv')

    # Feature Engineering
    df['content'] = df['kategori'] + ' ' + df['preferensi']

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['content'])

    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def get_recommendations(title, cosine_sim=cosine_sim, df=df):
        try:
            idx = df.index[df['nama'] == title].tolist()[0]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:11]
            place_indices = [i[0] for i in sim_scores]
            return df['nama'].iloc[place_indices]
        except IndexError:
            print(f"Place '{title}' not found in dataset.")
            return pd.Series()

    # Example recommendation
    recommended_places = get_recommendations('Pantai Mengening')
    print("Recommended places:", recommended_places)

    # Save models
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
    joblib.dump(cosine_sim, 'cosine_sim.joblib')

    # MLflow logging
    with mlflow.start_run(description="Content-Based Recommender Model") as run:
        # Log parameters
        mlflow.log_param("vectorizer_type", "TF-IDF")
        mlflow.log_param("similarity_metric", "cosine")
        
        # Log metrics
        mlflow.log_metric("num_recommendations", len(recommended_places))
        mlflow.log_metric("vocabulary_size", len(tfidf_vectorizer.vocabulary_))

        # Log artifacts
        mlflow.log_artifact('tfidf_vectorizer.joblib')
        mlflow.log_artifact('cosine_sim.joblib')
        mlflow.log_artifact('dataset_tempat_wisata_bali_processed.csv')

except Exception as e:
    print(f"An error occurred: {str(e)}")
