import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

os.makedirs("saved_models", exist_ok=True)
print("Training and exporting models to saved_models/...")

# ----------------------------------------------------
# 1. Hospital Length of Stay (LOS) Models
# ----------------------------------------------------
df_los = pd.read_csv(os.path.join("Classical ML Time-Series", "dummy_hospital_generated.csv"))

numerical_cols = df_los.select_dtypes(include=["int64", "float64"]).columns
for col in numerical_cols:
    if df_los[col].isnull().sum() > 0:
        df_los[col] = df_los[col].fillna(df_los[col].median())

categorical_cols = df_los.select_dtypes(include=["object", "category"]).columns
for col in categorical_cols:
    if df_los[col].isnull().sum() > 0:
        df_los[col] = df_los[col].fillna(df_los[col].mode()[0])

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_los[col] = le.fit_transform(df_los[col].astype(str))
    encoders[col] = le

cols_to_drop = ["Case_No", "DoctorLicense", "DoctorName"]
for col in cols_to_drop:
    if col in df_los.columns:
        df_los.drop(columns=[col], inplace=True)

X_los = df_los.drop(columns=["LOS"])
y_los = df_los["LOS"]

# Train Random Forest & XGBoost
rf_los = RandomForestRegressor(n_estimators=100, random_state=42)
rf_los.fit(X_los, y_los)

xgb_los = XGBRegressor(n_estimators=100, random_state=42)
xgb_los.fit(X_los, y_los)

joblib.dump(rf_los, "saved_models/los_rf_model.joblib")
joblib.dump(xgb_los, "saved_models/los_xgb_model.joblib")
joblib.dump(encoders, "saved_models/los_encoders.joblib")
joblib.dump(list(X_los.columns), "saved_models/los_feature_names.joblib")

print("LOS Models exported successfully!")

# ----------------------------------------------------
# 2. Network Anomaly Detection Models (NSL-KDD)
# ----------------------------------------------------
columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty_level"
]

train_file = os.path.join("Security Detection", "nsl-kdd", "KDDTrain+.txt")
df_sec = pd.read_csv(train_file, names=columns)
df_sec.dropna(inplace=True)
df_sec["binary_label"] = (df_sec["label"] != "normal").astype(int)
df_sec.drop(columns=["label", "difficulty_level"], inplace=True)

num_cols_sec = list(df_sec.select_dtypes(include=["int64", "float64"]).columns.drop("binary_label"))
X_sec_num = df_sec[num_cols_sec]
y_sec = df_sec["binary_label"]

sec_scaler = StandardScaler()
X_sec_scaled = sec_scaler.fit_transform(X_sec_num)

# Train Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(X_sec_scaled)

# Train Autoencoder on normal data
X_sec_normal = X_sec_scaled[y_sec == 0]
input_dim = X_sec_scaled.shape[1]

input_layer = Input(shape=(input_dim,))
encoder = Dense(32, activation="relu")(input_layer)
encoder = Dense(16, activation="relu")(encoder)
decoder = Dense(32, activation="relu")(encoder)
decoder = Dense(input_dim, activation="linear")(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(X_sec_normal, X_sec_normal, epochs=10, batch_size=256, verbose=0)

# Threshold at 95th percentile
pred_normal = autoencoder.predict(X_sec_normal)
normal_errors = np.mean(np.square(X_sec_normal - pred_normal), axis=1)
threshold = float(np.percentile(normal_errors, 95))

joblib.dump(iso_forest, "saved_models/sec_iso_forest.joblib")
joblib.dump(sec_scaler, "saved_models/sec_scaler.joblib")
joblib.dump(num_cols_sec, "saved_models/sec_feature_names.joblib")
autoencoder.save("saved_models/sec_autoencoder.keras")

with open("saved_models/sec_threshold.json", "w") as f:
    json.dump({"threshold": threshold}, f)

print(f"Security Models exported! Threshold: {threshold:.4f}")
print("ALL MODELS SAVED SUCCESSFULLY!")
