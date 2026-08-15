import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)
CORS(app)

print("Loading saved ML models into memory...")
MODELS_DIR = "saved_models"

# 1. Load LOS Models & Encoders
rf_los = joblib.load(os.path.join(MODELS_DIR, "los_rf_model.joblib"))
xgb_los = joblib.load(os.path.join(MODELS_DIR, "los_xgb_model.joblib"))
los_encoders = joblib.load(os.path.join(MODELS_DIR, "los_encoders.joblib"))
los_feature_names = joblib.load(os.path.join(MODELS_DIR, "los_feature_names.joblib"))

# 2. Load Security Anomaly Models
iso_forest = joblib.load(os.path.join(MODELS_DIR, "sec_iso_forest.joblib"))
sec_scaler = joblib.load(os.path.join(MODELS_DIR, "sec_scaler.joblib"))
sec_feature_names = joblib.load(os.path.join(MODELS_DIR, "sec_feature_names.joblib"))
autoencoder = tf.keras.models.load_model(os.path.join(MODELS_DIR, "sec_autoencoder.keras"))

with open(os.path.join(MODELS_DIR, "sec_threshold.json"), "r") as f:
    sec_threshold = json.load(f)["threshold"]

print(f"Models loaded successfully! Security threshold: {sec_threshold:.4f}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/predict_los", methods=["POST"])
def predict_los():
    try:
        data = request.json or {}
        row = {}
        for feature in los_feature_names:
            val = data.get(feature, 0)
            if feature in los_encoders:
                le = los_encoders[feature]
                str_val = str(val)
                if str_val in le.classes_:
                    encoded = le.transform([str_val])[0]
                else:
                    # Pick first available class or transform nearest match
                    encoded = 0
                row[feature] = float(encoded)
            else:
                try:
                    row[feature] = float(val)
                except (ValueError, TypeError):
                    row[feature] = 0.0

        input_df = pd.DataFrame([row], columns=los_feature_names)
        
        rf_pred = float(rf_los.predict(input_df)[0])
        xgb_pred = float(xgb_los.predict(input_df)[0])
        ensemble_pred = (rf_pred + xgb_pred) / 2.0

        risk_category = "Low Stay (< 3 Days)"
        if ensemble_pred >= 6.0:
            risk_category = "Extended Stay (> 6 Days)"
        elif ensemble_pred >= 3.5:
            risk_category = "Moderate Stay (3-6 Days)"

        return jsonify({
            "status": "success",
            "rf_prediction": round(rf_pred, 2),
            "xgb_prediction": round(xgb_pred, 2),
            "ensemble_prediction": round(ensemble_pred, 2),
            "risk_category": risk_category
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/predict_anomaly", methods=["POST"])
def predict_anomaly():
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

        # Isolation Forest prediction (-1: anomaly, 1: normal)
        if_raw = iso_forest.predict(scaled_input)[0]
        if_is_anomaly = bool(if_raw == -1)

        # Autoencoder prediction
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

from waitress import serve

if __name__ == "__main__":
    print("================================================================")
    print("MITACS Portfolio Production WSGI Server (Powered by Waitress)")
    print("Hospital LOS Predictor & Network Anomaly Threat Suite")
    print("Serving Production Web App on http://localhost:5000")
    print("================================================================")
    serve(app, host="0.0.0.0", port=5000, threads=8)


