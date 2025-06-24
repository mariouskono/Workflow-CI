import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
import dagshub
import joblib
import os
import sklearn
import sys # Import sys untuk keluar jika ada kesalahan fatal

print("✅ Starting script...")

# === Secure DagsHub Token Configuration ===
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise ValueError("DAGSHUB_TOKEN environment variable is not set.")

DAGSHUB_TRACKING_URI = 'https://dagshub.com/mariouskono/modelll.mlflow'
LOCAL_TRACKING_URI = "file:./mlruns" # Jalur tracking MLflow lokal

# Coba untuk menginisialisasi Dagshub dan mengatur URI tracking remote
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

try:
    # Logging MLflow dimulai di sini
    print("📊 Memulai MLflow run...")
    # mlflow.start_run akan menggunakan URI tracking yang diatur dalam os.environ['MLFLOW_TRACKING_URI']
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

        # Fungsi Rekomendasi
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

        # Simpan artefak non-MLflow (ini harus tetap disimpan bahkan jika logging MLflow gagal)
        joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
        joblib.dump(cosine_sim, 'cosine_sim.joblib')
        print("✅ Artefak non-MLflow disimpan secara lokal.")

        # Selalu log model dan artefak dalam konteks MLflow run
        print("📊 Melakukan logging artefak dan model ke MLflow (lokal atau remote)...")
        mlflow.log_param("vectorizer_type", "TF-IDF")
        mlflow.log_param("similarity_metric", "cosine")
        mlflow.log_metric("num_recommendations", len(recommended_places))
        mlflow.log_metric("vocabulary_size", len(tfidf_vectorizer.vocabulary_))

        mlflow.log_artifact('cosine_sim.joblib')
        mlflow.log_artifact('dataset_tempat_wisata_bali_processed.csv')

        mlflow.sklearn.log_model(
            sk_model=tfidf_vectorizer,
            artifact_path="tfidf_model",
            registered_model_name="TFIDFRecommender"
        )
        print("✅ Model dan artefak MLflow berhasil dilog.")

    print("✅ MLflow run (bagian lokal) selesai.")
except Exception as e:
    print(f"❌ Terjadi error selama eksekusi MLflow run: {str(e)}")
    # Jika MLflow run utama gagal, kita perlu memastikan pipeline CI tahu itu gagal.
    sys.exit(1) # Keluar dengan kode error
