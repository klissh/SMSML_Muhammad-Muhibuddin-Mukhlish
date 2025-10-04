# Import library yang dibutuhkan
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import mlflow
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# --- 1. Konfigurasi MLflow & DagsHub (Gabungan) ---
print("Initializing Combined MLflow (Local + DagsHub)...")

# WAJIB: Atur URI pelacakan ke server LOKAL terlebih dahulu
# Ini memenuhi syarat Basic/Skilled
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Inisialisasi DagsHub, ini akan "membungkus" URI lokal dan me-mirror ke DagsHub
# Ini memenuhi syarat Advanced
dagshub.init(repo_owner='klissh', repo_name='proyek-akhir-mlops', mlflow=True)

# Mulai sesi MLflow. Log akan dikirim ke URI lokal DAN DagsHub
mlflow.start_run()
print("MLflow run started. Logging to both local server and DagsHub.")

# --- 2. Memuat dan Mempersiapkan Data ---
print("Loading and preparing data...")
df = pd.read_csv('dataset_preprocessing/creditcard_processed.csv')
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data preparation complete.")

# --- 3. Hyperparameter Tuning ---
print("Starting hyperparameter tuning...")
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print("Hyperparameter tuning complete. Best params found:", best_params)

# --- 4. Evaluasi Model ---
print("Evaluating the best model...")
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# --- 5. Manual Logging ---
print("Logging parameters, metrics, and artifacts...")
mlflow.log_params(best_params)
mlflow.log_metric("accuracy", accuracy)

# Artefak 1: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
confusion_matrix_path = "confusion_matrix.png"
plt.savefig(confusion_matrix_path)
mlflow.log_artifact(confusion_matrix_path, "evaluation_results")
os.remove(confusion_matrix_path)

# Artefak 2: Classification Report
report = classification_report(y_test, y_pred, output_dict=True)
report_path = "classification_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=4)
mlflow.log_artifact(report_path, "evaluation_results")
os.remove(report_path)

# --- 6. Log Model ---
mlflow.sklearn.log_model(best_model, "random_forest_model")
print("Model has been logged.")

mlflow.end_run()
print("MLflow run finished successfully.")