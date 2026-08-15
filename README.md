# MITACS Multi-Project Machine Learning & Cybersecurity Portfolio (2027)

Welcome to the **MITACS Modular ML Portfolio**, structured as an expandable multi-project architecture where each project is a self-contained microservice folder with its own model training pipeline, saved artifacts, and dedicated web application server.

---

## 📁 Projects Included

### 1. 🏥 `01_hospital_los_prediction/` (Port 5001)
- **Goal:** Predict patient hospital Length of Stay (LOS) in days using clinical and administrative features.
- **Models:** Linear Regression, Decision Tree, Random Forest, XGBoost.
- **Winning Model:** **Random Forest Regressor** (MAE = `0.2436` days, RMSE = `0.5691` days).
- **Web App:** Self-contained Flask/Waitress production app listening on **`http://localhost:5001`**.

### 2. 🛡️ `02_network_anomaly_detection/` (Port 5002)
- **Goal:** Direct IP Address Threat Lookup & Cyberattack Anomaly Detection (NSL-KDD dataset).
- **Features:** Direct IP Threat Lookup (`/inspect_ip`), Tor Exit Node detection, Subnet Reputation, Isolation Forest, Deep Bottleneck Autoencoders (TensorFlow/Keras).
- **Winning Model:** **Deep Autoencoder** (Macro F1 = `82.86%`, Threshold = `0.1419`).
- **Web App:** Self-contained Flask/Waitress production app listening on **`http://localhost:5002`**.

---

## ☁️ How to Deploy on Render.com

This repository includes a ready-to-use **Render Blueprint (`render.yaml`)**, `.env.example`, and `Procfile`s for automatic deployment.

### **Method 1: Automatic Blueprint Deployment (Recommended)**
1. Push your repository to **GitHub**.
2. Log into [Render.com](https://render.com) and click **New +** -> **Blueprint**.
3. Connect your GitHub repository `mitacs_portfolio_2027`.
4. Render will automatically detect `render.yaml` and create 2 production Web Services:
   - **`hospital-los-predictor`**
   - **`agentguard-anomaly-detector`**
5. Click **Apply**. Render will install requirements, train/export models, and start the WSGI server (`gunicorn`).

---

### **Method 2: Manual Web Service Deployment on Render**
For deploying **Project 2 (Network Anomaly & IP Threat Detector)** as a standalone service:
* **Environment**: `Python 3`
* **Build Command**: `pip install -r requirements.txt && python 02_network_anomaly_detection/train_and_export.py`
* **Start Command**: `gunicorn --chdir 02_network_anomaly_detection app:app`

For deploying **Project 1 (Hospital LOS Predictor)** as a standalone service:
* **Environment**: `Python 3`
* **Build Command**: `pip install -r requirements.txt && python 01_hospital_los_prediction/train_and_export.py`
* **Start Command**: `gunicorn --chdir 01_hospital_los_prediction app:app`

---

## 💻 Local Execution

```powershell
# 1. Activate Environment
.\mitacs\Scripts\activate

# 2. Run Master Launcher
python run_portfolio.py
```
* Select **`1`**: Runs Hospital LOS Predictor on `http://localhost:5001`
* Select **`2`**: Runs AgentGuard Anomaly & IP Threat Detector on `http://localhost:5002`
* Select **`A`**: Runs **ALL projects in parallel**

---

## 🌐 API Reference (Project 2 IP Inspection)

### **Inspect IP Address Threat**
* **Endpoint**: `POST /inspect_ip`
* **Payload**: `{"ip_address": "185.220.101.5"}`
* **Response**:
```json
{
  "status": "success",
  "ip_address": "185.220.101.5",
  "threat_level": "CRITICAL THREAT (95%+ RISK)",
  "reputation": "Malicious IP / Known Attack Origin",
  "isolation_forest_anomaly": true,
  "autoencoder_anomaly": true,
  "reconstruction_error": 0.4743,
  "threshold": 0.1419
}
```
