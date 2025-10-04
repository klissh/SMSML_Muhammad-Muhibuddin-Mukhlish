# Import library yang dibutuhkan
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import dagshub

# --- 1. Inisialisasi DagsHub & MLflow ---
print("Initializing DagsHub and MLflow for basic model...")
# Ganti dengan username dan nama repo DagsHub-mu
dagshub.init(repo_owner='klissh', repo_name='proyek-akhir-mlops', mlflow=True)

# Set nama eksperimen yang berbeda untuk memisahkan hasil
mlflow.set_experiment("Credit Scoring Basic Model")

# --- 2. Aktifkan MLflow Autologging ---
# Autolog akan secara otomatis mencatat parameter, metrik, dan model
mlflow.autolog()
print("MLflow autologging enabled.")

# --- 3. Memuat dan Mempersiapkan Data ---
print("Loading and preparing data...")
df = pd.read_csv('dataset_preprocessing/creditcard_processed.csv')

# Pisahkan fitur (X) dan target (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Bagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data preparation complete.")

# --- 4. Melatih Model Sederhana ---
# Mulai sesi MLflow run. Autolog akan bekerja di dalam blok ini.
with mlflow.start_run() as run:
    print("Training a simple RandomForestClassifier model...")
    # Definisikan model dengan parameter sederhana
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    
    # Latih model
    model.fit(X_train, y_train)
    print("Model training complete.")
    
    # Evaluasi model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy of the basic model: {accuracy:.4f}")
    print(f"Run ID: {run.info.run_id}")

print("Basic model experiment finished. Check your DagsHub repository!")