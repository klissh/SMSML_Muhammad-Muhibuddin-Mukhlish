import os
import sys
import time
import logging
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import mlflow
import mlflow.sklearn
from datetime import datetime
# Make MLflow and joblib optional to avoid startup failures
try:
    import mlflow
    import mlflow.sklearn
except Exception:
    mlflow = None
try:
    import joblib
except Exception:
    joblib = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('ml_inference_requests_total', 'Total inference requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('ml_inference_request_duration_seconds', 'Request latency')
PREDICTION_CONFIDENCE = Gauge('ml_prediction_confidence', 'Last prediction confidence score')
MODEL_LOAD_TIME = Gauge('ml_model_load_time_seconds', 'Time taken to load the model')
ACTIVE_CONNECTIONS = Gauge('ml_active_connections', 'Number of active connections')
ERROR_COUNT = Counter('ml_inference_errors_total', 'Total inference errors', ['error_type'])

# ===== Added metrics required by assignment (http_*) =====
# Counter for total HTTP requests (instrumented per request)
HTTP_REQUESTS_TOTAL = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
# Histogram base name "http_request_duration_second" will generate _sum/_count/_bucket series automatically
HTTP_REQUEST_DURATION_SECOND = Histogram('http_request_duration_second', 'HTTP request duration in seconds')
# Counter to represent completed HTTP requests (can be used with rate() for throughput)
HTTP_REQUEST_THROUGHPUT_TOTAL = Counter('http_request_throughput_total', 'Total completed HTTP requests (use rate() for throughput)')
# ===== end added metrics =====

app = Flask(__name__)

# Quick helper: check if a host:port is reachable (to skip long MLflow retries)
import socket

def is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        s.close()
        return True
    except Exception:
        try:
            s.close()
        except Exception:
            pass
        return False

# Fallback simple model when MLflow/Sklearn is unavailable
class SimpleThresholdModel:
    def predict(self, X):
        arr = np.array(X)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return (arr.sum(axis=1) > 0).astype(int)

class CreditScoringModel:
    def __init__(self):
        self.model = None
        self.model_version = None
        self.model_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            start_time = time.time()
            loaded_from = None
            # Try to load from MLflow first only if server reachable
            if mlflow is not None and is_port_open("localhost", 5000, 0.5):
                try:
                    mlflow.set_tracking_uri("http://localhost:5000")
                    model_name = "credit-scoring-model"
                    model_version = "latest"
                    model_uri = f"models:/{model_name}/{model_version}"
                    self.model = mlflow.sklearn.load_model(model_uri)
                    self.model_version = model_version
                    loaded_from = f"MLflow: {model_uri}"
                    logger.info(f"Model loaded from MLflow: {model_uri}")
                except Exception as mlflow_error:
                    logger.warning(f"Could not load from MLflow: {mlflow_error}")
            else:
                if mlflow is None:
                    logger.info("MLflow not installed; skipping MLflow loading")
                else:
                    logger.info("MLflow server not reachable on localhost:5000; skipping MLflow loading")
            
            # Fallback: try to load from local file using joblib
            if self.model is None and joblib is not None:
                try:
                    import glob
                    model_path = "../Membangun_model/mlruns/0/*/artifacts/model/model.pkl"
                    model_files = glob.glob(model_path)
                    if model_files:
                        latest_model = max(model_files, key=os.path.getctime)
                        self.model = joblib.load(latest_model)
                        self.model_version = "local"
                        loaded_from = f"local file: {latest_model}"
                        logger.info(f"Model loaded from local file: {latest_model}")
                except Exception as e:
                    logger.warning(f"Could not load local model: {e}")
            
            # Final fallback: simple threshold or dummy sklearn
            if self.model is None:
                try:
                    try:
                        from sklearn.ensemble import RandomForestClassifier
                        from sklearn.datasets import make_classification
                        X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                        model.fit(X, y)
                        self.model = model
                        self.model_version = "dummy_sklearn"
                        loaded_from = "dummy sklearn"
                        logger.info("Using dummy sklearn model for demonstration")
                    except Exception:
                        self.model = SimpleThresholdModel()
                        self.model_version = "simple_threshold"
                        loaded_from = "simple threshold"
                        logger.info("Using SimpleThresholdModel for demonstration")
                except Exception as e:
                    logger.error(f"Failed to build fallback model: {e}")
            
            load_time = time.time() - start_time
            MODEL_LOAD_TIME.set(load_time)
            self.model_loaded = True
            logger.info(f"Model loaded successfully from {loaded_from} in {load_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            ERROR_COUNT.labels(error_type='model_load_error').inc()
            raise
    
    def predict(self, features):
        """Make prediction with the loaded model"""
        if not self.model_loaded:
            raise ValueError("Model not loaded")
        
        try:
            # Convert features to numpy array if needed
            if isinstance(features, list):
                features = np.array(features).reshape(1, -1)
            elif isinstance(features, dict):
                # Convert dict to array (assuming ordered features)
                features = np.array(list(features.values())).reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            
            # Get prediction probability if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)[0]
                confidence = max(probabilities)
                PREDICTION_CONFIDENCE.set(confidence)
            else:
                confidence = 0.8  # Default confidence
                PREDICTION_CONFIDENCE.set(confidence)
            
            return {
                'prediction': int(prediction),
                'confidence': float(confidence),
                'model_version': self.model_version,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            ERROR_COUNT.labels(error_type='prediction_error').inc()
            raise

# Initialize model
model_service = CreditScoringModel()

@app.before_request
def before_request():
    """Track active connections and increment per-request counters"""
    ACTIVE_CONNECTIONS.inc()
    # Populate http_requests_total with method and endpoint labels
    try:
        HTTP_REQUESTS_TOTAL.labels(method=request.method, endpoint=request.path).inc()
    except Exception:
        # Fail-safe to avoid crashing if request context is missing
        pass

@app.after_request
def after_request(response):
    """Track request completion"""
    ACTIVE_CONNECTIONS.dec()
    # Increment throughput counter on completed response
    HTTP_REQUEST_THROUGHPUT_TOTAL.inc()
    return response

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Measure request duration for this endpoint
    with HTTP_REQUEST_DURATION_SECOND.time():
        REQUEST_COUNT.labels(method='GET', endpoint='/health').inc()
        return jsonify({
            'status': 'healthy',
            'model_loaded': model_service.model_loaded,
            'model_version': model_service.model_version,
            'timestamp': datetime.now().isoformat()
        })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    # Measure both custom and assignment-required duration metrics
    with REQUEST_LATENCY.time(), HTTP_REQUEST_DURATION_SECOND.time():
        REQUEST_COUNT.labels(method='POST', endpoint='/predict').inc()
        
        try:
            # Get request data
            data = request.get_json()
            
            if not data:
                ERROR_COUNT.labels(error_type='invalid_input').inc()
                return jsonify({'error': 'No data provided'}), 400
            
            # Extract features
            if 'features' in data:
                features = data['features']
            else:
                # Assume the entire data is features
                features = data
            
            # Make prediction
            result = model_service.predict(features)
            
            logger.info(f"Prediction made: {result}")
            return jsonify(result)
            
        except Exception as e:
            ERROR_COUNT.labels(error_type='general_error').inc()
            logger.error(f"Prediction endpoint error: {e}")
            return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    # Measure both custom and assignment-required duration metrics
    with REQUEST_LATENCY.time(), HTTP_REQUEST_DURATION_SECOND.time():
        REQUEST_COUNT.labels(method='POST', endpoint='/predict/batch').inc()
        
        try:
            data = request.get_json()
            
            if not data or 'instances' not in data:
                ERROR_COUNT.labels(error_type='invalid_input').inc()
                return jsonify({'error': 'No instances provided'}), 400
            
            instances = data['instances']
            results = []
            
            for instance in instances:
                try:
                    result = model_service.predict(instance)
                    results.append(result)
                except Exception as e:
                    ERROR_COUNT.labels(error_type='batch_prediction_error').inc()
                    results.append({'error': str(e)})
            
            return jsonify({'predictions': results})
            
        except Exception as e:
            ERROR_COUNT.labels(error_type='general_error').inc()
            logger.error(f"Batch prediction error: {e}")
            return jsonify({'error': str(e)}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    # Measure request duration for this endpoint
    with HTTP_REQUEST_DURATION_SECOND.time():
        REQUEST_COUNT.labels(method='GET', endpoint='/model/info').inc()
        return jsonify({
            'model_loaded': model_service.model_loaded,
            'model_version': model_service.model_version,
            'model_type': type(model_service.model).__name__ if model_service.model else None,
            'timestamp': datetime.now().isoformat()
        })

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.errorhandler(404)
def not_found(error):
    ERROR_COUNT.labels(error_type='not_found').inc()
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    ERROR_COUNT.labels(error_type='internal_error').inc()
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting Credit Scoring Inference Service...")
    logger.info("Available endpoints:")
    logger.info("  GET  /health - Health check")
    logger.info("  POST /predict - Single prediction")
    logger.info("  POST /predict/batch - Batch predictions")
    logger.info("  GET  /model/info - Model information")
    logger.info("  GET  /metrics - Prometheus metrics")
    
    # Start the Flask application
    app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)