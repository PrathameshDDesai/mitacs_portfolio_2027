import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

print("[02_network_anomaly_detection] Training and exporting models...")

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

train_file = os.path.join(BASE_DIR, "nsl-kdd", "KDDTrain+.txt")
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

# Calculate 95th Percentile Threshold
pred_normal = autoencoder.predict(X_sec_normal)
normal_errors = np.mean(np.square(X_sec_normal - pred_normal), axis=1)
threshold = float(np.percentile(normal_errors, 95))

# Save artifacts inside Project 2 folder
joblib.dump(iso_forest, os.path.join(SAVED_MODELS_DIR, "iso_forest.joblib"))
joblib.dump(sec_scaler, os.path.join(SAVED_MODELS_DIR, "scaler.joblib"))
joblib.dump(num_cols_sec, os.path.join(SAVED_MODELS_DIR, "feature_names.joblib"))
autoencoder.save(os.path.join(SAVED_MODELS_DIR, "autoencoder.keras"))

with open(os.path.join(SAVED_MODELS_DIR, "threshold.json"), "w") as f:
    json.dump({"threshold": threshold}, f)

print(f"[02_network_anomaly_detection] Models exported successfully! Threshold: {threshold:.4f}")
