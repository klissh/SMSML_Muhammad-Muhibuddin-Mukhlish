import time
import random
import psutil
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Summary
import threading
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Prometheus metrics
# Model performance metrics
model_accuracy = Gauge('ml_model_accuracy', 'Current model accuracy')
model_precision = Gauge('ml_model_precision', 'Current model precision')
model_recall = Gauge('ml_model_recall', 'Current model recall')
model_f1_score = Gauge('ml_model_f1_score', 'Current model F1 score')

# Inference metrics
inference_requests_total = Counter('ml_inference_requests_total', 'Total number of inference requests')
inference_duration = Histogram('ml_inference_duration_seconds', 'Time spent on inference')
inference_errors_total = Counter('ml_inference_errors_total', 'Total number of inference errors')

# System metrics
cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
memory_usage = Gauge('system_memory_usage_percent', 'Memory usage percentage')
disk_usage = Gauge('system_disk_usage_percent', 'Disk usage percentage')

# Model specific metrics
model_predictions_total = Counter('ml_model_predictions_total', 'Total number of predictions made')
model_confidence = Summary('ml_model_confidence', 'Model prediction confidence scores')
model_latency = Histogram('ml_model_latency_seconds', 'Model prediction latency')

# Data drift metrics
data_drift_score = Gauge('ml_data_drift_score', 'Data drift detection score')
feature_importance = Gauge('ml_feature_importance', 'Feature importance scores', ['feature_name'])

class MetricsCollector:
    def __init__(self):
        self.running = True
        
    def collect_model_metrics(self):
        """Simulate model performance metrics collection"""
        while self.running:
            try:
                # Simulate model performance metrics (in real scenario, these would come from actual model evaluation)
                model_accuracy.set(random.uniform(0.85, 0.95))
                model_precision.set(random.uniform(0.80, 0.90))
                model_recall.set(random.uniform(0.75, 0.85))
                model_f1_score.set(random.uniform(0.78, 0.88))
                
                # Simulate data drift
                data_drift_score.set(random.uniform(0.0, 0.3))
                
                # Simulate feature importance for top features
                features = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
                for feature in features:
                    feature_importance.labels(feature_name=feature).set(random.uniform(0.1, 0.9))
                
                time.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"Error collecting model metrics: {e}")
                time.sleep(5)
    
    def collect_system_metrics(self):
        """Collect system performance metrics"""
        while self.running:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_usage.set(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_usage.set(memory.percent)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                disk_usage.set(disk_percent)
                
                time.sleep(10)  # Update every 10 seconds
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                time.sleep(5)
    
    def simulate_inference_metrics(self):
        """Simulate inference request metrics"""
        while self.running:
            try:
                # Simulate inference requests
                inference_requests_total.inc()
                
                # Simulate inference duration
                duration = random.uniform(0.1, 2.0)
                inference_duration.observe(duration)
                
                # Simulate model latency
                latency = random.uniform(0.05, 0.5)
                model_latency.observe(latency)
                
                # Simulate prediction confidence
                confidence = random.uniform(0.6, 0.99)
                model_confidence.observe(confidence)
                
                # Increment predictions counter
                model_predictions_total.inc()
                
                # Occasionally simulate errors
                if random.random() < 0.05:  # 5% error rate
                    inference_errors_total.inc()
                
                time.sleep(random.uniform(1, 5))  # Random interval between requests
            except Exception as e:
                logger.error(f"Error simulating inference metrics: {e}")
                time.sleep(5)
    
    def start_collection(self):
        """Start all metric collection threads"""
        threads = [
            threading.Thread(target=self.collect_model_metrics, daemon=True),
            threading.Thread(target=self.collect_system_metrics, daemon=True),
            threading.Thread(target=self.simulate_inference_metrics, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
        
        return threads

def main():
    """Main function to start the Prometheus exporter"""
    logger.info("Starting Prometheus metrics exporter...")
    
    # Start Prometheus HTTP server
    start_http_server(8000)
    logger.info("Prometheus metrics server started on port 8000")
    
    # Initialize and start metrics collector
    collector = MetricsCollector()
    threads = collector.start_collection()
    
    logger.info("Metrics collection started")
    logger.info("Metrics available at http://localhost:8000/metrics")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down metrics exporter...")
        collector.running = False
        
        # Wait for threads to finish
        for thread in threads:
            thread.join(timeout=5)
        
        logger.info("Metrics exporter stopped")

if __name__ == '__main__':
    main()