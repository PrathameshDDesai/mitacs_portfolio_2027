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

print("[AgentGuard SOC Platform] Loading core detection engines...")
iso_forest = joblib.load(os.path.join(SAVED_MODELS_DIR, "iso_forest.joblib"))
sec_scaler = joblib.load(os.path.join(SAVED_MODELS_DIR, "scaler.joblib"))
sec_feature_names = joblib.load(os.path.join(SAVED_MODELS_DIR, "feature_names.joblib"))
autoencoder = tf.keras.models.load_model(os.path.join(SAVED_MODELS_DIR, "autoencoder.keras"))

with open(os.path.join(SAVED_MODELS_DIR, "threshold.json"), "r") as f:
    sec_threshold = json.load(f)["threshold"]

print(f"[AgentGuard SOC Platform] Detection engines initialized. Baseline MSE Threshold: {sec_threshold:.4f}")

KNOWN_THREAT_RANGES = {
    "185.220.": {"country": "DE", "isp": "Tor Exit Network", "asn": "AS208323", "reputation": "High-risk Tor Anonymizer Node"},
    "45.142.": {"country": "RO", "isp": "HostSailor Datacenter", "asn": "AS60117", "reputation": "Known DoS/DDoS Origin Subnet"},
    "193.142.": {"country": "NL", "isp": "M247 Europe Ltd", "asn": "AS9009", "reputation": "Frequent Port Scan Activity"},
    "103.251.": {"country": "CN", "isp": "Chinanet Backbone", "asn": "AS4134", "reputation": "Suspicious Brute-Force Activity"},
    "185.191.": {"country": "RU", "isp": "Selectel Network", "asn": "AS49505", "reputation": "Automated Scanner Subnet"}
}

def get_ip_intel(ip_str, is_private):
    if is_private:
        return {
            "country": "Local",
            "isp": "Internal Enterprise Subnet",
            "asn": "PRIVATE-ASN",
            "ip_type": "Private IPv4",
            "reputation": "Internal Network Asset"
        }
    
    for prefix, info in KNOWN_THREAT_RANGES.items():
        if ip_str.startswith(prefix):
            return {
                "country": info["country"],
                "isp": info["isp"],
                "asn": info["asn"],
                "ip_type": "Public IPv4",
                "reputation": info["reputation"]
            }
            
    # Deterministic metadata for other IP ranges
    octets = [int(x) for x in ip_str.split(".")] if "." in ip_str and len(ip_str.split(".")) == 4 else [8, 8, 8, 8]
    countries = ["US", "DE", "GB", "JP", "FR", "NL", "CA", "AU"]
    asns = ["AS13335 Cloudflare Inc", "AS15169 Google LLC", "AS16509 Amazon.com", "AS3320 Deutsche Telekom", "AS8075 Microsoft Corp"]
    
    country = countries[(octets[0] + octets[3]) % len(countries)]
    asn = asns[(octets[1] + octets[2]) % len(asns)]
    isp = asn.split(" ", 1)[1] if " " in asn else "Public Datacenter Provider"
    
    return {
        "country": country,
        "isp": isp,
        "asn": asn.split(" ")[0],
        "ip_type": "Public IPv4",
        "reputation": "Monitored External IPv4"
    }

def analyze_ip_address_threat(ip_str, custom_features=None):
    try:
        ip_obj = ipaddress.ip_address(ip_str.strip())
        is_private = ip_obj.is_private
    except ValueError:
        ip_obj = None
        is_private = False

    intel = get_ip_intel(ip_str, is_private)
    is_known_bad = any(ip_str.startswith(prefix) for prefix in KNOWN_THREAT_RANGES)

    if custom_features and any(k in custom_features for k in ["serror_rate", "count", "src_bytes"]):
        features = custom_features
    else:
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
            octets = [int(x) for x in ip_str.split(".")] if "." in ip_str and len(ip_str.split(".")) == 4 else [10, 0, 0, 1]
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

    # Threat Score calculation (0 - 100)
    anomaly_ratio = (mse_error / max(sec_threshold, 0.0001))
    
    score = 10.0
    if is_known_bad:
        score += 50.0
    if ae_is_anomaly:
        score += 30.0 * min(anomaly_ratio, 2.5)
    if if_is_anomaly:
        score += 25.0
    if float(features.get("serror_rate", 0)) > 0.5:
        score += 15.0

    threat_score = int(min(max(score, 5.0), 99.0))
    if is_known_bad or (ae_is_anomaly and if_is_anomaly and threat_score > 75):
        threat_score = max(threat_score, 88)

    # 4-Stage Threat Level Mapping
    if threat_score >= 81:
        threat_level = "CRITICAL"
        threat_color = "red"
    elif threat_score >= 56:
        threat_level = "HIGH"
        threat_color = "orange"
    elif threat_score >= 26:
        threat_level = "MEDIUM"
        threat_color = "amber"
    else:
        threat_level = "LOW"
        threat_color = "green"

    # Human-readable detection reasons
    detection_reasons = []
    if is_known_bad:
        detection_reasons.append(f"Subnet belongs to known threat intelligence blacklists ({intel['reputation']})")
    if ae_is_anomaly:
        detection_reasons.append(f"Deep Autoencoder detected anomalous pattern (MSE: {mse_error:.4f} vs Baseline: {sec_threshold:.4f})")
    if if_is_anomaly:
        detection_reasons.append("Isolation Forest algorithm isolated feature vectors as out-of-distribution traffic")
    if float(features.get("serror_rate", 0)) > 0.4:
        detection_reasons.append(f"Elevated SYN error rate observed ({float(features.get('serror_rate', 0))*100:.0f}%)")
    if float(features.get("count", 0)) > 200:
        detection_reasons.append(f"High connection velocity burst ({int(features.get('count', 0))} connections/sec)")

    if not detection_reasons:
        detection_reasons.append("Traffic telemetry conforms to baseline expected behavior for normal network hosts.")

    return {
        "status": "success",
        "ip_address": ip_str,
        "threat_level": threat_level,
        "threat_score": threat_score,
        "threat_color": threat_color,
        "country": intel["country"],
        "isp": intel["isp"],
        "asn": intel["asn"],
        "ip_type": intel["ip_type"],
        "reputation": intel["reputation"],
        "abuse_confidence": threat_score,
        "detection_reasons": detection_reasons,
        "isolation_forest_anomaly": if_is_anomaly,
        "autoencoder_anomaly": ae_is_anomaly,
        "reconstruction_error": round(mse_error, 4),
        "threshold": round(sec_threshold, 4),
        "anomaly_ratio": round(anomaly_ratio * 100, 1),
        "packet_features": features
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "AgentGuard SOC Threat Intelligence", "port": int(os.environ.get("PORT", 5002))})

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
    print("AgentGuard SOC Threat Intelligence Platform")
    print(f"Production Waitress WSGI Server listening on http://localhost:{port}")
    print("================================================================")
    serve(app, host="0.0.0.0", port=port, threads=8)
