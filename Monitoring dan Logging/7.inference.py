import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# Gunakan endpoint MLflow Serve untuk prediksi
MLFLOW_ENDPOINT = os.getenv("MLFLOW_ENDPOINT", "http://127.0.0.1:1234/invocations")

@app.route("/predict", methods=["POST"])
def predict():
    # Ambil JSON dari body request
    data = request.get_json(force=True)

    # Payload sesuai format MLflow pyfunc
    payload = {
        "dataframe_records": [data]
    }

    try:
        response = requests.post(
            MLFLOW_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=5.0
        )
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({"error": response.text}), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Jalankan Flask (default di port 5000)
    app.run(debug=True)