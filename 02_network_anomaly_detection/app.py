import os
import json
import joblib
import ipaddress
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

KNOWN_MALICIOUS_IP_PATTERNS = [
    "185.220.", "45.142.", "193.142.", "103.251.", "185.191."
]

def analyze_ip_address_threat(ip_str, custom_features=None):
    """
    Evaluates an IP address and inspects threat level based on subnet intelligence,
    IP hashing, and ML model inference (Isolation Forest + Deep Autoencoder).
    """
    try:
        ip_obj = ipaddress.ip_address(ip_str.strip())
        is_private = ip_obj.is_private
    except ValueError:
        ip_obj = None
        is_private = False

    # Check known threat ranges
    is_known_bad = any(ip_str.startswith(prefix) for prefix in KNOWN_MALICIOUS_IP_PATTERNS)

    if custom_features:
        features = custom_features
    else:
        # Generate feature signature based on IP characteristics
        if is_known_bad:
            features = {
                "duration": 0, "src_bytes": 0, "dst_bytes": 0,
                "count": 320, "srv_count": 320, "serror_rate": 1.0,
                "same_srv_rate": 1.0, "dst_host_count": 255
            }
        elif is_private:
            features = {
                "duration": 0, "src_bytes": 240, "dst_bytes": 3800,
                "count": 4, "srv_count": 4, "serror_rate": 0.0,
                "same_srv_rate": 1.0, "dst_host_count": 255
            }
        else:
            # Hash IP octets into packet simulation parameters
            octets = [int(x) for x in ip_str.split(".")] if "." in ip_str else [10, 0, 0, 1]
            serror = round(((octets[0] + octets[3]) % 100) / 100.0, 2)
            count_val = int((octets[2] * 3 + octets[3]) % 350)
            features = {
                "duration": int(octets[3] % 10),
                "src_bytes": int(octets[1] * 50),
                "dst_bytes": int(octets[2] * 120),
                "count": count_val,
                "srv_count": max(1, int(count_val * 0.8)),
                "serror_rate": serror,
                "same_srv_rate": round(1.0 - serror, 2),
                "dst_host_count": 255
            }

    row = []
    for feature in sec_feature_names:
        val = features.get(feature, 0.0)
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

    # Determine Threat Level Category
    if is_known_bad or (if_is_anomaly and ae_is_anomaly) or (mse_error > sec_threshold * 2.5):
        threat_level = "CRITICAL THREAT (95%+ RISK)"
        threat_color = "red"
        reputation = "Malicious IP / Known Attack Origin"
    elif if_is_anomaly or ae_is_anomaly:
        threat_level = "HIGH RISK (ANOMALOUS TRAFFIC)"
        threat_color = "red"
        reputation = "Suspicious Traffic Patterns Detected"
    elif mse_error > sec_threshold * 0.7:
        threat_level = "MODERATE RISK (ELEVATED ERROR)"
        threat_color = "amber"
        reputation = "Monitored / Unverified Host"
    else:
        threat_level = "CLEAN / BENIGN IP"
        threat_color = "green"
        reputation = "Trusted Network Host"

    return {
        "status": "success",
        "ip_address": ip_str,
        "threat_level": threat_level,
        "threat_color": threat_color,
        "reputation": reputation,
        "is_private_ip": is_private,
        "isolation_forest_anomaly": if_is_anomaly,
        "autoencoder_anomaly": ae_is_anomaly,
        "reconstruction_error": round(mse_error, 4),
        "threshold": round(sec_threshold, 4),
        "anomaly_ratio": round((mse_error / max(sec_threshold, 0.0001)) * 100, 1),
        "packet_features": features
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "project": "02_network_anomaly_detection", "port": int(os.environ.get("PORT", 5002))})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json or {}
        ip_addr = data.get("ip_address", "192.168.1.100")
        result = analyze_ip_address_threat(ip_addr, custom_features=data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/inspect_ip", methods=["POST"])
def inspect_ip():
    try:
        data = request.json or {}
        ip_addr = data.get("ip_address", "").strip()
        if not ip_addr:
            return jsonify({"status": "error", "message": "IP address is required"}), 400

        result = analyze_ip_address_threat(ip_addr)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    print("================================================================")
    print("Project 2: AgentGuard Network Anomaly Threat Detector Server")
    print(f"Production Waitress WSGI Server listening on http://localhost:{port}")
    print("================================================================")
    serve(app, host="0.0.0.0", port=port, threads=8)
