# 🚀 MITACS Multi-Project Machine Learning & Cybersecurity Portfolio (2027)

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Flask%20%7C%20Waitress%20%7C%20Gunicorn-green.svg)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost%20%7C%20Isolation%20Forest-orange.svg)
![Deployment](https://img.shields.io/badge/Deployment-Render.com-purple.svg)
![Status](https://img.shields.io/badge/Production-Live-success.svg)

A modular, enterprise-grade multi-project portfolio featuring real-world Machine Learning and Cybersecurity applications. Built with a scalable microservices architecture where each project is a self-contained web service folder complete with its own training pipelines, saved model artifacts, and dedicated web applications.

---

## 🌐 Live Production Deployments

| Project Name | Domain | Live Production URL | Port (Local) |
| :--- | :--- | :--- | :--- |
| **Project 1: Healthcare OS** | Clinical Decision Support & Bed Allocation | 🔗 [mitacs-portfolio-2027-2.onrender.com](https://mitacs-portfolio-2027-2.onrender.com/) | `5001` |
| **Project 2: AgentGuard SOC** | IP Threat Intelligence & Anomaly Operations | 🔗 [network-anomaly-detection3.onrender.com](https://network-anomaly-detection3.onrender.com/) | `5002` |

---

## 📁 Repository Structure

```text
mitacs_portfolio_2027/
├── 01_hospital_los_prediction/           # Project 1: Healthcare Length of Stay (LOS) Predictor
│   ├── app.py                            # Production Flask application (Port 5001)
│   ├── Procfile                          # Render Gunicorn deployment config
│   ├── requirements.txt                  # Subfolder dependencies
│   ├── train_and_export.py               # Model training script (Random Forest & XGBoost)
│   ├── saved_models/                     # Exported .joblib model artifacts & scalers
│   └── templates/index.html              # Medical Slate Clinical UI Platform
│
├── 02_network_anomaly_detection/         # Project 2: AgentGuard SOC Threat Intelligence
│   ├── app.py                            # Production Flask application (Port 5002)
│   ├── Procfile                          # Render Gunicorn deployment config
│   ├── requirements.txt                  # Subfolder dependencies
│   ├── train_and_export.py               # Anomaly model training script (Isolation Forest & Autoencoder)
│   ├── saved_models/                     # Exported .joblib and threshold.json artifacts
│   ├── nsl-kdd/                          # Benchmark network security dataset
│   └── templates/index.html              # SOC Dark Slate Threat Intelligence UI Platform
│
├── render.yaml                           # Master Render Blueprint deployment configuration
├── .python-version                       # Python 3.10.13 version pin
├── .env.example                          # Environment variable configuration template
├── run_portfolio.py                      # Interactive CLI launcher for local execution
└── README.md                             # Master documentation
```

---

## 🏥 Project 1: Healthcare OS — Hospital Length of Stay (LOS) Predictor

### 🎯 **Overview**
A Clinical Decision Support System designed for hospital administrators and attending physicians to predict patient Length of Stay (LOS) in days upon admission. Calculates ensemble model variance ($\pm$ Days), stay risk level (`LOW`, `MODERATE`, `EXTENDED`), bed allocation guidance (Standard Ward vs ICU), and actionable discharge notes.

### 🧠 **Machine Learning Pipeline**
- **Algorithms Evaluated**: Linear Regression, Decision Tree, Random Forest Regressor, XGBoost Regressor.
- **Top Performing Model**: **Random Forest Regressor** (MAE: `0.2436` days, RMSE: `0.5691` days).
- **Features Processed**: Case Mix Index (CMI), Case Severity Level, Specialty, Doctor License Type, Payor Plan, Hospital Revenue, Admission Month, Gender.

### 📊 **UI Features**
- **3-Stage Visual Stay Risk Meter**: `SHORT STAY (<3 Days)` $\rightarrow$ `MODERATE STAY (3-6 Days)` $\rightarrow$ `EXTENDED STAY (>6 Days)`.
- **Bed Resource Allocation Guidance**: Recommends Acute Care, Specialized Inpatient, or ICU Bed based on clinical severity.
- **Patient Risk Log**: Live browser-session memory logging all patient admission assessments.

---

## 🛡️ Project 2: AgentGuard SOC — Threat Intelligence & Network Anomaly Platform

### 🎯 **Overview**
An enterprise Security Operations Center (SOC) IP Threat Intelligence platform styled after AbuseIPDB, VirusTotal, and Splunk. Allows security analysts to query IPv4 addresses for threat scoring (0-100), 4-stage risk levels (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`), GeoIP/ASN metadata, and out-of-distribution anomaly detection.

### 🧠 **Machine Learning & Intelligence Pipeline**
- **Algorithms**: Isolation Forest (Scikit-Learn) + Deep Bottleneck Autoencoders (TensorFlow/Keras).
- **Dataset**: NSL-KDD Network Intrusion Benchmark Dataset.
- **Threat Intelligence Engines**:
  - `Tor Exit Node & Threat Subnet Lookup`: Instant identification of known high-risk anonymizer nodes and DoS subnets.
  - `Isolation Forest Anomaly Detection`: Out-of-distribution feature isolation.
  - `Deep Autoencoder Reconstruction MSE`: Baseline thresholding (`0.1419`).

### 📊 **UI Features**
- **Command-Line Style Search Bar**: Direct IP address inspection with preset threat target buttons (`Tor Exit`, `DoS Subnet`, `Port Scan`, `Google DNS`, `Local Host`).
- **4-Stage Visual Threat Meter**: Segmented progress bar highlighting threat levels from `LOW` (Green) to `CRITICAL` (Red).
- **2-Column Intelligence Grid**: IP metadata (Country, ISP, ASN, Reputation) alongside AI model diagnostics and human-readable detection reasons.

---

## 💻 Local Execution Guide

### **Prerequisites**
- Python 3.10+ installed
- Git installed

### **Setup & Execution**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/PrathameshDDesai/mitacs_portfolio_2027.git
   cd mitacs_portfolio_2027
   ```

2. **Activate Virtual Environment**:
   ```powershell
   # Windows PowerShell
   .\mitacs\Scripts\activate
   ```

3. **Launch the Master Portfolio Launcher**:
   ```bash
   python run_portfolio.py
   ```
   - Type **`1`**: Launches Project 1 (Healthcare OS) on `http://localhost:5001`
   - Type **`2`**: Launches Project 2 (AgentGuard SOC) on `http://localhost:5002`
   - Type **`A`**: Launches **both projects in parallel**

---

## ☁️ Deploying on Render.com

This repository includes a pre-configured **Render Blueprint (`render.yaml`)** for automatic cloud deployment.

### **Automatic Blueprint Deployment (Recommended)**
1. Log into your [Render.com Dashboard](https://render.com).
2. Click **New +** $\rightarrow$ **Blueprint**.
3. Connect your GitHub repository `PrathameshDDesai/mitacs_portfolio_2027`.
4. Render will automatically detect `render.yaml` and provision both Web Services:
   - **`hospital-los-predictor`**
   - **`agentguard-anomaly-detector`**
5. Click **Apply**. Render will run build commands, export models, and launch Gunicorn WSGI servers automatically.

---

## 🌐 API Specifications

### **Project 1: Predict Length of Stay**
- **Endpoint**: `POST /predict`
- **Request Body**:
  ```json
  {
    "CMI Value": 2.5,
    "Revenue": 25000,
    "Severity": "High",
    "Specialty": "General Surgery"
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "ensemble_prediction": 2.19,
    "rf_prediction": 2.22,
    "xgb_prediction": 2.16,
    "risk_category": "Low Stay (< 3 Days)"
  }
  ```

### **Project 2: Inspect IP Address Threat**
- **Endpoint**: `POST /inspect_ip`
- **Request Body**:
  ```json
  {
    "ip_address": "185.220.101.5"
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "ip_address": "185.220.101.5",
    "threat_level": "CRITICAL",
    "threat_score": 99,
    "country": "DE",
    "isp": "Tor Exit Network",
    "asn": "AS208323",
    "reputation": "High-risk Tor Anonymizer Node",
    "isolation_forest_anomaly": true
  }
  ```

---

## 📄 License & Attribution

Created as part of the **MITACS Machine Learning & Cybersecurity Portfolio (2027)** by **Prathamesh Desai**. Developed for high-performance clinical analytics and enterprise network security monitoring.
