import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
import dagshub
import joblib
import os
import sklearn

print("✅ Starting script...")

# === Secure DagsHub Token Configuration ===
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise ValueError("DAGSHUB_TOKEN environment variable is not set.")
# Dikoreksi: MLFLOW_TRACKING_URI harus menunjuk ke 'modelll' seperti yang dikonfirmasi pengguna
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/mariouskono/modelll.mlflow'

# Authenticate and initialize DagsHub
print("🔐 Authenticating with DagsHub...")
dagshub.auth.add_app_token(dagshub_token)
# Dikoreksi: repo_name harus 'modelll' seperti yang dikonfirmasi pengguna
dagshub.init(repo_owner='mariouskono', repo_name='modelll', mlflow=True)
print("✅ DagsHub authenticated and MLflow initialized.")

try:
    # Load dataset
    print("📥 Loading dataset...")
    df = pd.read_csv('dataset_tempat_wisata_bali_processed.csv')
    print("✅ Dataset loaded. Rows:", len(df))

    # Feature Engineering
    print("🧠 Creating 'content' column...")
    df['content'] = df['kategori'] + ' ' + df['preferensi']

    # TF-IDF Vectorization
    print("🔢 Vectorizing content...")
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['content'])

    # Cosine Similarity
    print("📐 Calculating similarity...")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Recommendation Function
    def get_recommendations(title, cosine_sim=cosine_sim, df=df):
        try:
            idx = df.index[df['nama'] == title].tolist()[0]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:11]
            place_indices = [i[0] for i in sim_scores]
            return df['nama'].iloc[place_indices]
        except IndexError:
            print(f"⚠️ Place '{title}' not found in dataset.")
            return pd.Series()

    print("🔍 Generating example recommendations...")
    recommended_places = get_recommendations('Pantai Mengening')
    print("✅ Recommendations:", recommended_places.tolist())

    # Save non-MLflow artifacts
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
    joblib.dump(cosine_sim, 'cosine_sim.joblib')

    # MLflow Logging
    print("📊 Logging to MLflow...")
    with mlflow.start_run(description="Content-Based Recommender Model") as run:
        mlflow.log_param("vectorizer_type", "TF-IDF")
        mlflow.log_param("similarity_metric", "cosine")
        mlflow.log_metric("num_recommendations", len(recommended_places))
        mlflow.log_metric("vocabulary_size", len(tfidf_vectorizer.vocabulary_))

        mlflow.log_artifact('cosine_sim.joblib')
        mlflow.log_artifact('dataset_tempat_wisata_bali_processed.csv')

        # ✅ Log a full MLflow model for Docker use
        mlflow.sklearn.log_model(
            sk_model=tfidf_vectorizer,
            artifact_path="tfidf_model",
            registered_model_name="TFIDFRecommender"
        )

        # Output run ID to file so the workflow can use it
        run_id = run.info.run_id
        with open("mlflow_run_id.txt", "w") as f:
            f.write(run_id)

    print("✅ MLflow run complete. Run ID:", run_id)

except Exception as e:
    print(f"❌ An error occurred: {str(e)}")
