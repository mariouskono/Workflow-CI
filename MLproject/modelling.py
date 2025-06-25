import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
import dagshub
import joblib
import os
import sys

print("✅ Starting script...")

dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise ValueError("DAGSHUB_TOKEN environment variable is not set.")

DAGSHUB_TRACKING_URI = 'https://dagshub.com/mariouskono/modelll.mlflow'
LOCAL_TRACKING_URI = "file:./mlruns"

# Try to initialize Dagshub and set remote tracking URI
# This outer try-except handles the Dagshub.init part
try:
    print("🔐 Authenticating with DagsHub...")
    dagshub.auth.add_app_token(dagshub_token)
    dagshub.init(repo_owner='mariouskono', repo_name='modelll', mlflow=True)
    os.environ['MLFLOW_TRACKING_URI'] = DAGSHUB_TRACKING_URI
    print("✅ Dagshub terautentikasi dan MLflow diinisialisasi untuk tracking remote.")
    remote_tracking_enabled = True
except Exception as e:
    print(f"⚠️ Gagal mengautentikasi atau menginisialisasi Dagshub untuk tracking remote: {str(e)}")
    print("Menggunakan tracking MLflow lokal sebagai gantinya.")
    os.environ['MLFLOW_TRACKING_URI'] = LOCAL_TRACKING_URI
    remote_tracking_enabled = False

print(f"URI Tracking MLflow diatur ke: {os.environ['MLFLOW_TRACKING_URI']}")

# Main script logic, ensuring model training and local logging always run
try:
    print("📊 Memulai MLflow run...")
    # mlflow.start_run will use the tracking URI set in os.environ['MLFLOW_TRACKING_URI']
    # If MLFLOW_TRACKING_URI is set to remote but remote is down, this might still error out.
    # The 'unsupported endpoint' error is coming from here.
    with mlflow.start_run(description="Content-Based Recommender Model") as run:
        # Load dataset
        print("📥 Memuat dataset...")
        df = pd.read_csv('dataset_tempat_wisata_bali_processed.csv')
        print("✅ Dataset dimuat. Baris:", len(df))

        # Feature Engineering
        print("🧠 Membuat kolom 'content'...")
        df['content'] = df['kategori'] + ' ' + df['preferensi']

        # TF-IDF Vectorization
        print("🔢 Melakukan vektorisasi konten...")
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['content'])

        # Cosine Similarity
        print("📐 Menghitung kemiripan...")
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
                print(f"⚠️ Tempat '{title}' tidak ditemukan di dataset.")
                return pd.Series()

        print("🔍 Menghasilkan contoh rekomendasi...")
        recommended_places = get_recommendations('Pantai Mengening')
        print("✅ Rekomendasi:", recommended_places.tolist())

        # Save non-MLflow artifacts (these should always be saved)
        try: # This inner try-except is for joblib saving issues
            joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
            joblib.dump(cosine_sim, 'cosine_sim.joblib')
            print("✅ Artefak non-MLflow disimpan secara lokal.")
        except Exception as joblib_e:
            print(f"❌ Gagal menyimpan artefak non-MLflow secara lokal: {str(joblib_e)}")
            sys.exit(1) # Critical failure if local saving fails

        # Attempt MLflow logging. This part can fail if remote is down.
        # This is where the 'unsupported endpoint' error manifests if remote_tracking_enabled is True.
        try:
            print("📊 Melakukan logging artefak dan model ke MLflow (lokal atau remote)...")
            mlflow.log_param("vectorizer_type", "TF-IDF")
            mlflow.log_param("similarity_metric", "cosine")
            mlflow.log_metric("num_recommendations", len(recommended_places))
            mlflow.log_metric("vocabulary_size", len(tfidf_vectorizer.vocabulary_))

            # Ensure model and data artifacts are logged whether remote or local
            mlflow.log_artifact('cosine_sim.joblib')
            mlflow.log_artifact('dataset_tempat_wisata_bali_processed.csv')

            mlflow.sklearn.log_model(
                sk_model=tfidf_vectorizer,
                artifact_path="tfidf_model", # This path is relative to the run's artifact URI
                registered_model_name="TFIDFRecommender" # This requires a connection to a model registry
            )
            print("✅ Model dan artefak MLflow berhasil dilog.")
        except Exception as mlflow_logging_e:
            # If remote logging fails, we print a warning but do NOT exit
            print(f"⚠️ Logging MLflow ke tracking server gagal: {str(mlflow_logging_e)}")
            print("Model dan artefak akan tetap tersedia di penyimpanan artefak lokal MLflow.")
            # Do NOT exit, let the run finish locally.
            # The model is still logged to the local mlruns directory because it's part of mlflow.start_run context

    print("✅ MLflow run (bagian lokal) selesai.")
except Exception as main_e:
    # This outer try-except catches errors in the core data processing or if mlflow.start_run fails even locally.
    print(f"❌ Terjadi error fatal selama eksekusi skrip: {str(main_e)}")
    sys.exit(1) # Exit if essential parts like data loading or local MLflow run setup fail
