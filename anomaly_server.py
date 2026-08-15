import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)
CORS(app)

MODELS_DIR = "saved_models"

print("Loading Security Anomaly Detection models for dedicated server...")
iso_forest = joblib.load(os.path.join(MODELS_DIR, "sec_iso_forest.joblib"))
sec_scaler = joblib.load(os.path.join(MODELS_DIR, "sec_scaler.joblib"))
sec_feature_names = joblib.load(os.path.join(MODELS_DIR, "sec_feature_names.joblib"))
autoencoder = tf.keras.models.load_model(os.path.join(MODELS_DIR, "sec_autoencoder.keras"))

with open(os.path.join(MODELS_DIR, "sec_threshold.json"), "r") as f:
    sec_threshold = json.load(f)["threshold"]

print(f"Dedicated Anomaly Server initialized! Safety threshold: {sec_threshold:.4f}")

ANOMALY_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgentGuard Network Anomaly Threat Server</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #090d16;
            --card-bg: rgba(30, 41, 59, 0.75);
            --card-border: rgba(255, 255, 255, 0.1);
            --accent-purple: #8b5cf6;
            --accent-blue: #3b82f6;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --text-primary: #f8fafc;
            --text-muted: #94a3b8;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', sans-serif; }
        body {
            background-color: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            background-image: 
                radial-gradient(at 100% 0%, rgba(139, 92, 246, 0.2) 0px, transparent 50%),
                radial-gradient(at 0% 100%, rgba(59, 130, 246, 0.2) 0px, transparent 50%);
            background-attachment: fixed;
            padding: 30px 20px;
        }

        .container { max-width: 1100px; margin: 0 auto; }
        header {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 30px; padding-bottom: 20px; border-bottom: 1px solid var(--card-border);
        }

        .title-group h1 { font-size: 26px; font-weight: 800; }
        .title-group p { font-size: 13px; color: var(--text-muted); }

        .server-badge {
            background: rgba(139, 92, 246, 0.2); border: 1px solid var(--accent-purple);
            color: var(--accent-purple); padding: 6px 14px; border-radius: 20px;
            font-size: 12px; font-weight: 700; font-family: 'JetBrains Mono', monospace;
        }

        .grid { display: grid; grid-template-columns: 6fr 6fr; gap: 24px; }
        .card {
            background: var(--card-bg); border: 1px solid var(--card-border);
            backdrop-filter: blur(16px); border-radius: 20px; padding: 24px;
        }

        .card-title { font-size: 18px; font-weight: 700; margin-bottom: 16px; display: flex; gap: 8px; align-items: center; }

        .preset-bar { display: flex; gap: 8px; margin-bottom: 16px; }
        .preset-btn {
            background: rgba(255, 255, 255, 0.05); border: 1px solid var(--card-border);
            color: #fff; padding: 6px 12px; border-radius: 8px; font-size: 12px; cursor: pointer;
        }
        .preset-btn:hover { background: rgba(255, 255, 255, 0.15); border-color: var(--accent-purple); }

        .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .form-group { display: flex; flex-direction: column; gap: 4px; }
        .form-group label { font-size: 12px; color: var(--text-muted); font-weight: 500; }
        .form-group input {
            background: rgba(15, 23, 42, 0.8); border: 1px solid var(--card-border);
            color: #fff; padding: 8px 12px; border-radius: 8px; font-size: 13px; outline: none;
        }
        .form-group input:focus { border-color: var(--accent-purple); }

        .btn-submit {
            width: 100%; background: linear-gradient(135deg, var(--accent-purple), var(--accent-blue));
            color: #fff; border: none; padding: 12px; border-radius: 10px; font-weight: 700;
            font-size: 14px; cursor: pointer; margin-top: 16px;
        }

        .result-panel { text-align: center; padding: 20px; background: rgba(15, 23, 42, 0.9); border-radius: 16px; border: 1px solid var(--card-border); }
        .threat-title { font-size: 24px; font-weight: 800; margin: 12px 0; }

        .metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 16px; }
        .metric-card { background: rgba(255, 255, 255, 0.03); border: 1px solid var(--card-border); padding: 12px; border-radius: 10px; }
        .metric-card h4 { font-size: 11px; color: var(--text-muted); }
        .metric-card p { font-size: 18px; font-weight: 700; margin-top: 4px; }

        .status-badge { display: inline-block; padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 12px; text-transform: uppercase; }
        .badge-green { background: rgba(16, 185, 129, 0.2); color: var(--accent-green); border: 1px solid var(--accent-green); }
        .badge-red { background: rgba(239, 68, 68, 0.2); color: var(--accent-red); border: 1px solid var(--accent-red); }

        .meter-bar { height: 10px; background: rgba(255, 255, 255, 0.1); border-radius: 5px; overflow: hidden; margin-top: 8px; }
        .meter-fill { height: 100%; width: 0%; background: linear-gradient(90deg, var(--accent-green), var(--accent-red)); transition: width 0.5s; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="title-group">
                <h1>🛡️ AgentGuard Threat Detection Server</h1>
                <p>Standalone Microservice Server running on Port 8000</p>
            </div>
            <div class="server-badge">SERVER: ONLINE (PORT 8000)</div>
        </header>

        <div class="grid">
            <div class="card">
                <div class="card-title">🌐 Packet Feature Inputs</div>
                <div class="preset-bar">
                    <button class="preset-btn" onclick="setPreset('normal')">Normal Web</button>
                    <button class="preset-btn" onclick="setPreset('dos')">Neptune DoS</button>
                    <button class="preset-btn" onclick="setPreset('portscan')">Satan Scan</button>
                </div>
                <form id="sec-form" onsubmit="submitThreat(event)">
                    <div class="form-grid">
                        <div class="form-group"><label>Duration</label><input type="number" id="duration" value="0"></div>
                        <div class="form-group"><label>Source Bytes</label><input type="number" id="src_bytes" value="215"></div>
                        <div class="form-group"><label>Dst Bytes</label><input type="number" id="dst_bytes" value="4500"></div>
                        <div class="form-group"><label>Count</label><input type="number" id="count" value="5"></div>
                        <div class="form-group"><label>Service Count</label><input type="number" id="srv_count" value="5"></div>
                        <div class="form-group"><label>SYN Error Rate</label><input type="number" step="0.01" id="serror_rate" value="0.0"></div>
                        <div class="form-group"><label>Same Service Rate</label><input type="number" step="0.01" id="same_srv_rate" value="1.0"></div>
                        <div class="form-group"><label>Dst Host Count</label><input type="number" id="dst_host_count" value="255"></div>
                    </div>
                    <button type="submit" class="btn-submit">⚡ Inspect Network Traffic</button>
                </form>
            </div>

            <div class="card">
                <div class="card-title">🔍 Real-Time Threat Analysis</div>
                <div class="result-panel">
                    <span id="badge" class="status-badge badge-green">Ready to Scan</span>
                    <div id="threat-title" class="threat-title">Awaiting Data</div>

                    <div class="metric-grid">
                        <div class="metric-card"><h4>Isolation Forest</h4><p id="if-val">--</p></div>
                        <div class="metric-card"><h4>Autoencoder</h4><p id="ae-val">--</p></div>
                    </div>

                    <div style="margin-top: 20px; text-align: left;">
                        <div style="display: flex; justify-content: space-between; font-size: 12px; color: var(--text-muted);">
                            <span>Reconstruction MSE Loss</span>
                            <span id="mse-val">0.0000</span>
                        </div>
                        <div class="meter-bar"><div id="meter-fill" class="meter-fill"></div></div>
                        <p style="font-size: 11px; color: var(--text-muted); margin-top: 6px;">Threshold: <span id="thresh-val">--</span></p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function setPreset(type) {
            if (type === 'normal') {
                document.getElementById('duration').value = '0';
                document.getElementById('src_bytes').value = '215';
                document.getElementById('dst_bytes').value = '4500';
                document.getElementById('count').value = '5';
                document.getElementById('srv_count').value = '5';
                document.getElementById('serror_rate').value = '0.0';
                document.getElementById('same_srv_rate').value = '1.0';
                document.getElementById('dst_host_count').value = '255';
            } else if (type === 'dos') {
                document.getElementById('duration').value = '0';
                document.getElementById('src_bytes').value = '0';
                document.getElementById('dst_bytes').value = '0';
                document.getElementById('count').value = '280';
                document.getElementById('srv_count').value = '280';
                document.getElementById('serror_rate').value = '1.0';
                document.getElementById('same_srv_rate').value = '1.0';
                document.getElementById('dst_host_count').value = '255';
            } else if (type === 'portscan') {
                document.getElementById('duration').value = '2';
                document.getElementById('src_bytes').value = '12';
                document.getElementById('dst_bytes').value = '0';
                document.getElementById('count').value = '150';
                document.getElementById('srv_count').value = '1';
                document.getElementById('serror_rate').value = '0.85';
                document.getElementById('same_srv_rate').value = '0.05';
                document.getElementById('dst_host_count').value = '150';
            }
        }

        async function submitThreat(e) {
            e.preventDefault();
            const payload = {
                duration: document.getElementById('duration').value,
                src_bytes: document.getElementById('src_bytes').value,
                dst_bytes: document.getElementById('dst_bytes').value,
                count: document.getElementById('count').value,
                srv_count: document.getElementById('srv_count').value,
                serror_rate: document.getElementById('serror_rate').value,
                same_srv_rate: document.getElementById('same_srv_rate').value,
                dst_host_count: document.getElementById('dst_host_count').value
            };

            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (data.status === 'success') {
                document.getElementById('threat-title').innerText = data.overall_threat;
                document.getElementById('if-val').innerText = data.isolation_forest_anomaly ? 'ANOMALY' : 'NORMAL';
                document.getElementById('ae-val').innerText = data.autoencoder_anomaly ? 'ANOMALY' : 'NORMAL';
                document.getElementById('mse-val').innerText = data.reconstruction_error;
                document.getElementById('thresh-val').innerText = data.threshold;

                const badge = document.getElementById('badge');
                const fill = document.getElementById('meter-fill');
                fill.style.width = Math.min(data.anomaly_ratio, 100) + '%';

                if (data.overall_threat.includes('ATTACK')) {
                    badge.className = 'status-badge badge-red';
                    badge.innerText = '⚠️ INTRUSION DETECTED';
                } else {
                    badge.className = 'status-badge badge-green';
                    badge.innerText = '✅ BENIGN TRAFFIC';
                }
            }
        }
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(ANOMALY_HTML)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "AgentGuard Anomaly Threat Server", "port": 8000})

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
    print("Starting Standalone AgentGuard Anomaly Server on http://localhost:8000")
    app.run(host="0.0.0.0", port=8000, debug=False)
