# MITACS Multi-Project Machine Learning & Cybersecurity Portfolio (2027)

Welcome to the **MITACS Modular ML Portfolio**, structured as an expandable multi-project architecture where each project is a self-contained microservice folder with its own model training pipeline, saved artifacts, and dedicated web application server.

---

## 📁 Projects Included

### 1. 🏥 `01_hospital_los_prediction/` (Port 5001)
- **Goal:** Predict patient hospital Length of Stay (LOS) in days using clinical and administrative features.
- **Models:** Linear Regression, Decision Tree, Random Forest, XGBoost.
- **Winning Model:** **Random Forest Regressor** (MAE = `0.2436` days, RMSE = `0.5691` days).
- **Web App:** Self-contained Flask/Waitress production app on **`http://localhost:5001`**.

### 2. 🛡️ `02_network_anomaly_detection/` (Port 5002)
- **Goal:** Detect cyberattacks and network intrusion attempts on the NSL-KDD benchmark. Inspired by **AgentGuard behavioral monitoring**.
- **Models:** Unsupervised Isolation Forest & Deep Bottleneck Autoencoders (TensorFlow/Keras).
- **Winning Model:** **Deep Autoencoder** (Macro F1 = `82.86%`, Threshold = `0.1419`).
- **Web App:** Self-contained Flask/Waitress production app on **`http://localhost:5002`**.

---

## ⚙️ Modular Multi-Project Architecture (Designed for 8+ Projects)

Each project follows a strict self-contained layout:
```
d:\mitacs_portfolio_2027\
├── 01_hospital_los_prediction/
│   ├── app.py                      # Standalone Waitress WSGI Web Server (Port 5001)
│   ├── train_and_export.py         # Model Training & Export Script
│   ├── 01_los_prediction.ipynb     # Interactive Jupyter Notebook
│   ├── saved_models/               # Exported Model Artifacts (.joblib)
│   └── templates/index.html        # Dedicated Web UI Template
│
├── 02_network_anomaly_detection/
│   ├── app.py                      # Standalone Waitress WSGI Web Server (Port 5002)
│   ├── train_and_export.py         # Model Training & Export Script
│   ├── 02_anomaly_detection.ipynb  # Interactive Jupyter Notebook
│   ├── saved_models/               # Exported Model Artifacts (.joblib, .keras)
│   └── templates/index.html        # Dedicated Web UI Template
│
├── run_portfolio.py                # Master Portfolio Launcher
└── requirements.txt
```

---

## 🚀 How to Run

### **Option 1: Launch via Master Portfolio Launcher**
```powershell
# Activate environment
.\mitacs\Scripts\activate

# Run Master Launcher
python run_portfolio.py
```
* Select **`1`** to run Hospital LOS Predictor on `http://localhost:5001`.
* Select **`2`** to run AgentGuard Anomaly Detector on `http://localhost:5002`.
* Select **`A`** to launch **ALL projects in parallel**.

### **Option 2: Launch Individual Projects Directly**

```powershell
# Run Project 1 (Hospital LOS Predictor)
python 01_hospital_los_prediction/app.py

# Run Project 2 (AgentGuard Network Anomaly Threat Detector)
python 02_network_anomaly_detection/app.py
```
