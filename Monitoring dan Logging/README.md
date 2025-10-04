# Monitoring dan Logging - Credit Scoring ML System

Sistem monitoring dan logging untuk model machine learning credit scoring menggunakan Prometheus dan Grafana.

## Struktur Folder

```
Monitoring dan Logging/
├── 1.bukti_serving                     # Dokumentasi bukti serving model
├── 2.prometheus.yml                    # Konfigurasi Prometheus
├── 3.prometheus_exporter.py           # Script untuk mengekspos metrics
├── 4.bukti monitoring Prometheus/     # Folder bukti monitoring Prometheus
│   ├── 1.monitoring_accuracy
│   ├── 2.monitoring_inference_requests
│   └── 3.monitoring_system_resources
├── 5.bukti monitoring Grafana/        # Folder bukti monitoring Grafana
│   ├── 1.monitoring_model_performance
│   ├── 2.monitoring_inference_metrics
│   ├── 3.monitoring_system_health
│   ├── 4.monitoring_data_drift
│   └── 5.monitoring_prediction_confidence
├── 6.bukti alerting Grafana/          # Folder bukti alerting Grafana
│   ├── 1.rules_high_error_rate
│   ├── 2.notifikasi_high_error_rate
│   ├── 3.rules_low_model_accuracy
│   ├── 4.notifikasi_low_model_accuracy
│   ├── 5.rules_high_data_drift
│   └── 6.notifikasi_high_data_drift
├── 7.inference.py                     # Script inference dengan monitoring
├── requirements.txt                   # Dependencies
└── README.md                          # Dokumentasi ini
```

## Setup dan Instalasi

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Prometheus
1. Download Prometheus dari https://prometheus.io/download/
2. Copy file `2.prometheus.yml` ke folder Prometheus
3. Jalankan Prometheus:
   ```bash
   ./prometheus --config.file=prometheus.yml
   ```
4. Akses Prometheus UI di http://localhost:9090

### 3. Setup Grafana
1. Download Grafana dari https://grafana.com/grafana/download
2. Install dan jalankan Grafana
3. Akses Grafana UI di http://localhost:3000 (admin/admin)
4. Tambahkan Prometheus sebagai data source (http://localhost:9090)
5. Buat dashboard dengan nama "Dashboard-rafyardhani"

### 4. Jalankan Monitoring System

#### Start Prometheus Exporter
```bash
python 3.prometheus_exporter.py
```
Metrics tersedia di http://localhost:8000/metrics

#### Start Inference Service
```bash
python 7.inference.py
```
API tersedia di http://localhost:5002

## Metrics yang Dimonitor

### Model Performance Metrics
- `ml_model_accuracy`: Akurasi model saat ini
- `ml_model_precision`: Precision model
- `ml_model_recall`: Recall model
- `ml_model_f1_score`: F1 score model

### Inference Metrics
- `ml_inference_requests_total`: Total inference requests
- `ml_inference_duration_seconds`: Durasi inference
- `ml_inference_errors_total`: Total error inference
- `ml_prediction_confidence`: Confidence score prediksi

### System Metrics
- `system_cpu_usage_percent`: Penggunaan CPU
- `system_memory_usage_percent`: Penggunaan memory
- `system_disk_usage_percent`: Penggunaan disk

### Data Quality Metrics
- `ml_data_drift_score`: Skor data drift
- `ml_feature_importance`: Importance features

## Alerting Rules

### 1. High Error Rate
- **Condition**: Error rate > 10% selama 1 menit
- **Severity**: Critical
- **Action**: Notifikasi ke tim engineering

### 2. Low Model Accuracy
- **Condition**: Accuracy < 80% selama 2 menit
- **Severity**: Warning
- **Action**: Notifikasi ke ML team

### 3. High Data Drift
- **Condition**: Drift score > 0.3 selama 3 menit
- **Severity**: Critical
- **Action**: Trigger model retraining

## Testing

### Test Inference API
```bash
# Health check
curl http://localhost:5002/health

# Single prediction
curl -X POST http://localhost:5002/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0, 4.0, 5.0]}'

# Batch prediction
curl -X POST http://localhost:5002/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"instances": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}'

# Model info
curl http://localhost:5002/model/info

# Metrics
curl http://localhost:5002/metrics
```

## Dashboard Grafana

Dashboard harus dibuat dengan nama **"Dashboard-rafyardhani"** sesuai username Dicoding.

### Panel yang Diperlukan:
1. **Model Performance**: Accuracy, Precision, Recall, F1 Score
2. **Inference Metrics**: Request rate, Latency, Error rate
3. **System Health**: CPU, Memory, Disk usage
4. **Data Quality**: Data drift score, Feature importance
5. **Prediction Confidence**: Confidence distribution

### Alert Rules:
1. High Error Rate (> 10%)
2. Low Model Accuracy (< 80%)
3. High Data Drift (> 0.3)

## Troubleshooting

### Prometheus tidak bisa scrape metrics
- Pastikan prometheus_exporter.py berjalan di port 8000
- Check konfigurasi targets di prometheus.yml
- Verify firewall settings

### Grafana tidak bisa connect ke Prometheus
- Pastikan Prometheus berjalan di port 9090
- Check data source configuration di Grafana
- Verify network connectivity

### Model tidak bisa di-load
- Check path model di inference.py
- Pastikan MLflow server berjalan (jika menggunakan MLflow)
- Verify model artifacts tersedia

## Catatan Penting

1. **Dashboard Name**: Harus menggunakan nama "Dashboard-rafyardhani" sesuai username Dicoding
2. **Screenshots**: Semua file bukti harus diganti dengan screenshot aktual saat implementasi
3. **Metrics**: Minimal 3 metrics untuk Basic, 5 untuk Skilled, 10 untuk Advanced
4. **Alerting**: Minimal 1 alert untuk Skilled, 3 alerts untuk Advanced