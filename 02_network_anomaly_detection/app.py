import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from waitress import serve

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

print("[Project 2: Network Anomaly Detection] Loading models...")
iso_forest = joblib.load(os.path.join(SAVED_MODELS_DIR, "iso_forest.joblib"))
sec_scaler = joblib.load(os.path.join(SAVED_MODELS_DIR, "scaler.joblib"))
sec_feature_names = joblib.load(os.path.join(SAVED_MODELS_DIR, "feature_names.joblib"))
autoencoder = tf.keras.models.load_model(os.path.join(SAVED_MODELS_DIR, "autoencoder.keras"))

with open(os.path.join(SAVED_MODELS_DIR, "threshold.json"), "r") as f:
    sec_threshold = json.load(f)["threshold"]

print(f"[Project 2: Network Anomaly Detection] Models loaded! Threshold: {sec_threshold:.4f}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "project": "02_network_anomaly_detection", "port": 5002})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json or {}
        row = []
        for feature in sec_feature_names:
            val = data.get(feature, 0.0)
            try:
                row.append(float(val))
            except (ValueError, TypeError):
                row.append(0.0)

        input_df = pd.DataFrame([row], columns=sec_feature_names)
        scaled_input = sec_scaler.transform(input_df)

        if_raw = iso_forest.predict(scaled_input)[0]
        if_is_anomaly = bool(if_raw == -1)

        reconstructed = autoencoder.predict(scaled_input)
        mse_error = float(np.mean(np.square(scaled_input - reconstructed)))
        ae_is_anomaly = bool(mse_error > sec_threshold)

        overall_threat = "ATTACK / ANOMALY DETECTED" if (if_is_anomaly or ae_is_anomaly) else "NORMAL TRAFFIC"

        return jsonify({
            "status": "success",
            "overall_threat": overall_threat,
            "isolation_forest_anomaly": if_is_anomaly,
            "autoencoder_anomaly": ae_is_anomaly,
            "reconstruction_error": round(mse_error, 4),
            "threshold": round(sec_threshold, 4),
            "anomaly_ratio": round((mse_error / max(sec_threshold, 0.0001)) * 100, 1)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    print("================================================================")
    print("Project 2: AgentGuard Network Anomaly Threat Detector Server")
    print("Production Waitress WSGI Server listening on http://localhost:5002")
    print("================================================================")
    serve(app, host="0.0.0.0", port=5002, threads=8)
