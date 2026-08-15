# MITACS Machine Learning & Cybersecurity Portfolio (2027)

Welcome to the **MITACS ML Portfolio**, demonstrating end-to-end Machine Learning development across healthcare time-series prediction and cybersecurity network anomaly detection.

---

## 📁 Projects Included

### 1. 🏥 [Classical ML Time-Series: Length of Stay (LOS) Prediction](./Classical%20ML%20Time-Series)
- **Goal:** Predict patient hospital Length of Stay (LOS) using clinical and administrative features.
- **Models:** Linear Regression, Decision Tree Regressor, Random Forest Regressor, XGBoost Regressor.
- **Key Outcome:** **XGBoost** achieved the best performance with **MAE = 0.2856** and **RMSE = 0.7009**.
- **Notebook:** [`01_los_prediction.ipynb`](./Classical%20ML%20Time-Series/01_los_prediction.ipynb)

### 2. 🛡️ [Security Detection: Anomaly Detection for Network Security](./Security%20Detection)
- **Goal:** Detect cyberattacks and network intrusion attempts on the NSL-KDD benchmark dataset. Inspired by **AgentGuard behavioral monitoring**.
- **Models:** Unsupervised Isolation Forest (`contamination=0.1`) & Deep Bottleneck Autoencoders (TensorFlow/Keras).
- **Key Outcome:** **Isolation Forest Macro F1 = 56.24%** | **Autoencoder Macro F1 = 82.86%**.
- **Notebook:** [`02_anomaly_detection.ipynb`](./Security%20Detection/02_anomaly_detection.ipynb)

---

## ⚙️ Environment Setup & Installation

```bash
# Clone Repository
git clone https://github.com/PrathameshDDesai/mitacs-ml-portfolio.git
cd mitacs-ml-portfolio

# Create virtual environment and install requirements
python -m venv mitacs
mitacs\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Repository Structure
```
mitacs_portfolio_2027/
├── Classical ML Time-Series/
│   ├── 01_los_prediction.ipynb
│   ├── dummy_hospital_generated.csv
│   ├── X_train.csv, X_test.csv, y_train.csv, y_test.csv
│   └── README.md
├── Security Detection/
│   ├── 02_anomaly_detection.ipynb
│   ├── nsl-kdd/
│   ├── X_train.csv, X_test.csv, y_train.csv, y_test.csv
│   └── README.md
├── requirements.txt
└── README.md
```
