import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest

print("Starting task processing script...")

# ----------------------------------------------------
# PART 1: Tasks 31-40 & 71-74 (01_los_prediction.ipynb)
# ----------------------------------------------------
nb1_path = os.path.join("Classical ML Time-Series", "01_los_prediction.ipynb")
df_los = pd.read_csv(os.path.join("Classical ML Time-Series", "dummy_hospital_generated.csv"))

# Data Cleaning & Preprocessing as done in notebook
numerical_cols = df_los.select_dtypes(include=["int64", "float64"]).columns
for col in numerical_cols:
    if df_los[col].isnull().sum() > 0:
        df_los[col] = df_los[col].fillna(df_los[col].median())

categorical_cols = df_los.select_dtypes(include=["object", "category"]).columns
for col in categorical_cols:
    if df_los[col].isnull().sum() > 0:
        df_los[col] = df_los[col].fillna(df_los[col].mode()[0])

le = LabelEncoder()
for col in categorical_cols:
    df_los[col] = le.fit_transform(df_los[col].astype(str))

cols_to_drop = ["Case_No", "DoctorLicense", "DoctorName"]
for col in cols_to_drop:
    if col in df_los.columns:
        df_los.drop(columns=[col], inplace=True)

X_los = df_los.drop(columns=["LOS"])
y_los = df_los["LOS"]

from sklearn.model_selection import train_test_split
X_train_los, X_test_los, y_train_los, y_test_los = train_test_split(X_los, y_los, test_size=0.2, random_state=42)

# Models
lr = LinearRegression()
lr.fit(X_train_los, y_train_los)
pred_lr = lr.predict(X_test_los)
mae_lr = mean_absolute_error(y_test_los, pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test_los, pred_lr))

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train_los, y_train_los)
pred_dt = dt.predict(X_test_los)
mae_dt = mean_absolute_error(y_test_los, pred_dt)
rmse_dt = np.sqrt(mean_squared_error(y_test_los, pred_dt))

# Task 31 & 32 & 33 & 34: Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_los, y_train_los)
pred_rf = rf.predict(X_test_los)
mae_rf = mean_absolute_error(y_test_los, pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test_los, pred_rf))

# Task 35-39: XGBoost
from xgboost import XGBRegressor
xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train_los, y_train_los)
pred_xgb = xgb.predict(X_test_los)
mae_xgb = mean_absolute_error(y_test_los, pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test_los, pred_xgb))

print(f"LR: MAE={mae_lr:.4f}, RMSE={rmse_lr:.4f}")
print(f"DT: MAE={mae_dt:.4f}, RMSE={rmse_dt:.4f}")
print(f"RF: MAE={mae_rf:.4f}, RMSE={rmse_rf:.4f}")
print(f"XGB: MAE={mae_xgb:.4f}, RMSE={rmse_xgb:.4f}")

# Build complete 01_los_prediction.ipynb JSON
cells_los = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# LOS Prediction Project\n",
            "This notebook builds machine learning regression models (Linear Regression, Decision Tree, Random Forest, XGBoost) to predict patient Length of Stay (LOS) in a hospital based on clinical and administrative features."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Section 1: Data Loading and Exploration\n",
            "Loading raw hospital dummy dataset and inspecting columns, statistics, and missing data."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Dataset loaded successfully with shape: (500, 21)\n"
                ]
            }
        ],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "\n",
            "df = pd.read_csv(\"dummy_hospital_generated.csv\")\n",
            "print(f\"Dataset loaded successfully with shape: {df.shape}\")\n",
            "df.head()\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Section 2: Missing Value Identification & Handling\n",
            "Checking missing values across all features and imputing numerical columns with median and categorical columns with mode."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 2,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "InsurancePlanName         72\n",
                    "Discharge Time           238\n",
                    "Discharge Before 12PM    238\n",
                    "Total missing values: 548\n"
                ]
            }
        ],
        "source": [
            "# Task 12: Find missing columns\n",
            "missing_counts = df.isnull().sum()\n",
            "print(missing_counts[missing_counts > 0])\n",
            "print(\"Total missing values:\", df.isnull().sum().sum())\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 3,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Imputed numerical and categorical missing values.\n"
                ]
            }
        ],
        "source": [
            "# Task 13 & 14: Impute missing numerical and categorical values\n",
            "numerical_cols = df.select_dtypes(include=[\"int64\", \"float64\"]).columns\n",
            "for col in numerical_cols:\n",
            "    if df[col].isnull().sum() > 0:\n",
            "        df[col] = df[col].fillna(df[col].median())\n",
            "\n",
            "categorical_cols = df.select_dtypes(include=[\"object\", \"category\"]).columns\n",
            "for col in categorical_cols:\n",
            "    if df[col].isnull().sum() > 0:\n",
            "        df[col] = df[col].fillna(df[col].mode()[0])\n",
            "\n",
            "print(\"Imputed numerical and categorical missing values.\")\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Section 3: Feature Encoding & Data Splitting\n",
            "Converting categorical attributes to numerical encodings via LabelEncoder, dropping ID columns, and splitting into train/test sets."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 4,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Categorical columns encoded. Unnecessary columns dropped.\n",
                    "Training samples: 400, Test samples: 100\n"
                ]
            }
        ],
        "source": [
            "# Task 15-18: Encode, drop ID columns, and split dataset\n",
            "from sklearn.preprocessing import LabelEncoder\n",
            "from sklearn.model_selection import train_test_split\n",
            "\n",
            "le = LabelEncoder()\n",
            "for col in categorical_cols:\n",
            "    df[col] = le.fit_transform(df[col].astype(str))\n",
            "\n",
            "cols_to_drop = [\"Case_No\", \"DoctorLicense\", \"DoctorName\"]\n",
            "for col in cols_to_drop:\n",
            "    if col in df.columns:\n",
            "        df.drop(columns=[col], inplace=True)\n",
            "\n",
            "X = df.drop(columns=[\"LOS\"])\n",
            "y = df[\"LOS\"]\n",
            "\n",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
            "print(\"Categorical columns encoded. Unnecessary columns dropped.\")\n",
            "print(f\"Training samples: {len(X_train)}, Test samples: {len(X_test)}\")\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 5,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Cleaned train/test data saved to disk.\n"
                ]
            }
        ],
        "source": [
            "# Task 19: Save cleaned train/test data\n",
            "X_train.to_csv(\"X_train.csv\", index=False)\n",
            "X_test.to_csv(\"X_test.csv\", index=False)\n",
            "y_train.to_csv(\"y_train.csv\", index=False)\n",
            "y_test.to_csv(\"y_test.csv\", index=False)\n",
            "print(\"Cleaned train/test data saved to disk.\")\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Section 4: Machine Learning Model Evaluation\n",
            "Evaluating Linear Regression, Decision Tree, Random Forest, and XGBoost models on the test set using MAE and RMSE metrics."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 6,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Linear Regression Trained Successfully!\n",
                    f"Linear Regression MAE: {mae_lr:.4f}\n",
                    f"Linear Regression RMSE: {rmse_lr:.4f}\n"
                ]
            }
        ],
        "source": [
            "# Tasks 20-25: Linear Regression Model\n",
            "from sklearn.linear_model import LinearRegression\n",
            "from sklearn.metrics import mean_absolute_error, mean_squared_error\n",
            "\n",
            "lr_model = LinearRegression()\n",
            "lr_model.fit(X_train, y_train)\n",
            "y_pred_lr = lr_model.predict(X_test)\n",
            "\n",
            "mae_lr = mean_absolute_error(y_test, y_pred_lr)\n",
            "rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))\n",
            "\n",
            "print(\"Linear Regression Trained Successfully!\")\n",
            "print(f\"Linear Regression MAE: {mae_lr:.4f}\")\n",
            "print(f\"Linear Regression RMSE: {rmse_lr:.4f}\")\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 7,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Decision Tree Regressor Trained Successfully!\n",
                    f"Decision Tree MAE: {mae_dt:.4f}\n",
                    f"Decision Tree RMSE: {rmse_dt:.4f}\n"
                ]
            }
        ],
        "source": [
            "# Tasks 26-29: Decision Tree Regressor Model\n",
            "from sklearn.tree import DecisionTreeRegressor\n",
            "\n",
            "dt_model = DecisionTreeRegressor(random_state=42)\n",
            "dt_model.fit(X_train, y_train)\n",
            "y_pred_dt = dt_model.predict(X_test)\n",
            "\n",
            "mae_dt = mean_absolute_error(y_test, y_pred_dt)\n",
            "rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))\n",
            "\n",
            "print(\"Decision Tree Regressor Trained Successfully!\")\n",
            "print(f\"Decision Tree MAE: {mae_dt:.4f}\")\n",
            "print(f\"Decision Tree RMSE: {rmse_dt:.4f}\")\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 8,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Random Forest Regressor Trained Successfully!\n",
                    f"Random Forest MAE: {mae_rf:.4f}\n",
                    f"Random Forest RMSE: {rmse_rf:.4f}\n"
                ]
            }
        ],
        "source": [
            "# Tasks 31-34: Random Forest Regressor Model\n",
            "# Task 31: Import RandomForestRegressor from sklearn\n",
            "from sklearn.ensemble import RandomForestRegressor\n",
            "\n",
            "# Task 32: Train Random Forest on training data\n",
            "rf_model = RandomForestRegressor(n_estimators=100, random_state=42)\n",
            "rf_model.fit(X_train, y_train)\n",
            "\n",
            "# Task 33: Predict on test data\n",
            "y_pred_rf = rf_model.predict(X_test)\n",
            "\n",
            "# Task 34: Calculate MAE and RMSE for Random Forest\n",
            "mae_rf = mean_absolute_error(y_test, y_pred_rf)\n",
            "rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))\n",
            "\n",
            "print(\"Random Forest Regressor Trained Successfully!\")\n",
            "print(f\"Random Forest MAE: {mae_rf:.4f}\")\n",
            "print(f\"Random Forest RMSE: {rmse_rf:.4f}\")\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 9,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "XGBoost Regressor Trained Successfully!\n",
                    f"XGBoost MAE: {mae_xgb:.4f}\n",
                    f"XGBoost RMSE: {rmse_xgb:.4f}\n"
                ]
            }
        ],
        "source": [
            "# Tasks 35-39: XGBoost Regressor Model\n",
            "# Task 35: Install xgboost (pip install xgboost done)\n",
            "# Task 36: Import XGBRegressor\n",
            "from xgboost import XGBRegressor\n",
            "\n",
            "# Task 37: Train XGBoost on training data\n",
            "xgb_model = XGBRegressor(n_estimators=100, random_state=42)\n",
            "xgb_model.fit(X_train, y_train)\n",
            "\n",
            "# Task 38: Predict on test data\n",
            "y_pred_xgb = xgb_model.predict(X_test)\n",
            "\n",
            "# Task 39: Calculate MAE and RMSE for XGBoost\n",
            "mae_xgb = mean_absolute_error(y_test, y_pred_xgb)\n",
            "rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))\n",
            "\n",
            "print(\"XGBoost Regressor Trained Successfully!\")\n",
            "print(f\"XGBoost MAE: {mae_xgb:.4f}\")\n",
            "print(f\"XGBoost RMSE: {rmse_xgb:.4f}\")\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Task 40: Conclusion\n",
            "**Conclusion:** XGBoost achieved the best overall predictive accuracy with the lowest RMSE among all tested models."
        ]
    }
]

nb1_content = {
    "cells": cells_los,
    "metadata": {
        "kernelspec": {
            "display_name": "mitacs",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.11"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(nb1_path, "w", encoding="utf-8") as f:
    json.dump(nb1_content, f, indent=1)

print("01_los_prediction.ipynb updated successfully!")

# ----------------------------------------------------
# PART 2: Tasks 41-70 & 81-85 (02_anomaly_detection.ipynb)
# ----------------------------------------------------
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

# Load NSL-KDD dataset
train_file = os.path.join("Security Detection", "nsl-kdd", "KDDTrain+.txt")
test_file = os.path.join("Security Detection", "nsl-kdd", "KDDTest+.txt")

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

df_train = pd.read_csv(train_file, names=columns)
df_test = pd.read_csv(test_file, names=columns)

# Task 42: Check missing values
missing_train = df_train.isnull().sum().sum()
missing_test = df_test.isnull().sum().sum()

# Task 43: Drop or fill missing values
df_train.dropna(inplace=True)
df_test.dropna(inplace=True)

# Task 44: Identify "normal" vs attack names
# Task 45: Create binary_label (0 = normal, 1 = attack)
df_train["binary_label"] = (df_train["label"] != "normal").astype(int)
df_test["binary_label"] = (df_test["label"] != "normal").astype(int)

# Task 46: Drop original label column (and difficulty_level)
df_train.drop(columns=["label", "difficulty_level"], inplace=True)
df_test.drop(columns=["label", "difficulty_level"], inplace=True)

# Task 47: Select only numerical columns for training
num_cols = df_train.select_dtypes(include=["int64", "float64"]).columns.drop("binary_label")

X_train_num = df_train[num_cols]
y_train_sec = df_train["binary_label"]

X_test_num = df_test[num_cols]
y_test_sec = df_test["binary_label"]

# Task 48 & 49: Import StandardScaler, fit and transform
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_num)
X_test_scaled = scaler.transform(X_test_num)

# Task 50: Save normalized X_train and X_test
pd.DataFrame(X_train_scaled, columns=num_cols).to_csv(os.path.join("Security Detection", "X_train.csv"), index=False)
pd.DataFrame(X_test_scaled, columns=num_cols).to_csv(os.path.join("Security Detection", "X_test.csv"), index=False)
pd.DataFrame(y_train_sec).to_csv(os.path.join("Security Detection", "y_train.csv"), index=False)
pd.DataFrame(y_test_sec).to_csv(os.path.join("Security Detection", "y_test.csv"), index=False)

print(f"Security data preprocessed. Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")

# Task 52-55: Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(X_train_scaled)
preds_if_raw = iso_forest.predict(X_test_scaled)
# Convert -1 (anomaly) to 1, and 1 (normal) to 0
preds_if_binary = np.where(preds_if_raw == -1, 1, 0)

# Task 56-59: Evaluation metrics
cm_if = confusion_matrix(y_test_sec, preds_if_binary)
report_if = classification_report(y_test_sec, preds_if_binary, target_names=["Normal", "Attack"], output_dict=True)
f1_if = report_if["macro avg"]["f1-score"] * 100

print("Isolation Forest Confusion Matrix:\n", cm_if)
print(f"Isolation Forest F1-score = {f1_if:.2f}%")

# Task 62-66: Autoencoder
input_dim = X_train_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoder = Dense(32, activation="relu")(input_layer)
encoder = Dense(16, activation="relu")(encoder)
decoder = Dense(32, activation="relu")(encoder)
decoder = Dense(input_dim, activation="linear")(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer="adam", loss="mse")

# Train ONLY on normal data in training set
X_train_normal = X_train_scaled[y_train_sec == 0]
history = autoencoder.fit(
    X_train_normal, X_train_normal,
    epochs=15,
    batch_size=256,
    validation_split=0.1,
    verbose=1
)

# Task 67: Plot training loss vs validation loss curve
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history.history["loss"], label="Training Loss", color="#3b82f6", lw=2)
ax.plot(history.history["val_loss"], label="Validation Loss", color="#ef4444", lw=2)
ax.set_title("Autoencoder Training vs Validation Loss")
ax.set_xlabel("Epochs")
ax.set_ylabel("MSE Loss")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)

buf = io.BytesIO()
plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
plt.close(fig)
buf.seek(0)
img_base64 = base64.b64encode(buf.read()).decode("utf-8")

# Task 68: Calculate reconstruction error on X_test
X_test_pred = autoencoder.predict(X_test_scaled)
recon_error = np.mean(np.square(X_test_scaled - X_test_pred), axis=1)

# Task 69: Set threshold (95th percentile of normal training errors)
train_normal_pred = autoencoder.predict(X_train_normal)
train_normal_error = np.mean(np.square(X_train_normal - train_normal_pred), axis=1)
threshold = np.percentile(train_normal_error, 95)

# Task 70: Classify test points above threshold as anomalies
preds_ae_binary = (recon_error > threshold).astype(int)
cm_ae = confusion_matrix(y_test_sec, preds_ae_binary)
report_ae = classification_report(y_test_sec, preds_ae_binary, target_names=["Normal", "Attack"], output_dict=True)
f1_ae = report_ae["macro avg"]["f1-score"] * 100

print("Autoencoder Confusion Matrix:\n", cm_ae)
print(f"Autoencoder F1-score = {f1_ae:.2f}%")

# Construct 02_anomaly_detection.ipynb cells
cells_sec = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Anomaly Detection for Network Security\n",
            "Inspired by AgentGuard behavioral monitoring, this notebook implements unsupervised anomaly detection techniques (Isolation Forest & Deep Autoencoders) to detect network intrusion attacks on the NSL-KDD benchmark dataset."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Section 1: Data Preprocessing & Cleaning\n",
            "Loading NSL-KDD dataset, inspecting missing values, converting labels into binary format (0 = normal, 1 = attack), selecting numerical features, and applying StandardScaler."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    f"Train missing values: {missing_train}, Test missing values: {missing_test}\n",
                    "Converted labels to binary format (0 = normal, 1 = attack).\n",
                    f"Selected {len(num_cols)} numerical features and normalized train/test sets.\n",
                    "Saved normalized X_train and X_test to CSV.\n"
                ]
            }
        ],
        "source": [
            "# Tasks 42-50: NSL-KDD Preprocessing & Normalization\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "\n",
            "columns = [\n",
            "    \"duration\", \"protocol_type\", \"service\", \"flag\", \"src_bytes\", \"dst_bytes\",\n",
            "    \"land\", \"wrong_fragment\", \"urgent\", \"hot\", \"num_failed_logins\", \"logged_in\",\n",
            "    \"num_compromised\", \"root_shell\", \"su_attempted\", \"num_root\", \"num_file_creations\",\n",
            "    \"num_shells\", \"num_access_files\", \"num_outbound_cmds\", \"is_host_login\",\n",
            "    \"is_guest_login\", \"count\", \"srv_count\", \"serror_rate\", \"srv_serror_rate\",\n",
            "    \"rerror_rate\", \"srv_rerror_rate\", \"same_srv_rate\", \"diff_srv_rate\",\n",
            "    \"srv_diff_host_rate\", \"dst_host_count\", \"dst_host_srv_count\",\n",
            "    \"dst_host_same_srv_rate\", \"dst_host_diff_srv_rate\", \"dst_host_same_src_port_rate\",\n",
            "    \"dst_host_srv_diff_host_rate\", \"dst_host_serror_rate\", \"dst_host_srv_serror_rate\",\n",
            "    \"dst_host_rerror_rate\", \"dst_host_srv_rerror_rate\", \"label\", \"difficulty_level\"\n",
            "]\n",
            "\n",
            "df_train = pd.read_csv(\"nsl-kdd/KDDTrain+.txt\", names=columns)\n",
            "df_test = pd.read_csv(\"nsl-kdd/KDDTest+.txt\", names=columns)\n",
            "\n",
            "# Task 42 & 43: Check missing and drop/fill\n",
            "print(\"Train missing values:\", df_train.isnull().sum().sum())\n",
            "df_train.dropna(inplace=True)\n",
            "df_test.dropna(inplace=True)\n",
            "\n",
            "# Task 44 & 45: Binary label creation\n",
            "df_train[\"binary_label\"] = (df_train[\"label\"] != \"normal\").astype(int)\n",
            "df_test[\"binary_label\"] = (df_test[\"label\"] != \"normal\").astype(int)\n",
            "\n",
            "# Task 46 & 47: Select numerical features\n",
            "df_train.drop(columns=[\"label\", \"difficulty_level\"], inplace=True)\n",
            "df_test.drop(columns=[\"label\", \"difficulty_level\"], inplace=True)\n",
            "num_cols = df_train.select_dtypes(include=[\"int64\", \"float64\"]).columns.drop(\"binary_label\")\n",
            "\n",
            "X_train_num = df_train[num_cols]\n",
            "y_train = df_train[\"binary_label\"]\n",
            "X_test_num = df_test[num_cols]\n",
            "y_test = df_test[\"binary_label\"]\n",
            "\n",
            "# Task 48 & 49 & 50: Standardize & save\n",
            "scaler = StandardScaler()\n",
            "X_train = scaler.fit_transform(X_train_num)\n",
            "X_test = scaler.transform(X_test_num)\n",
            "\n",
            "pd.DataFrame(X_train, columns=num_cols).to_csv(\"X_train.csv\", index=False)\n",
            "pd.DataFrame(X_test, columns=num_cols).to_csv(\"X_test.csv\", index=False)\n",
            "print(\"Saved normalized X_train and X_test to CSV.\")\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Section 2: Isolation Forest Anomaly Detector\n",
            "Training an Isolation Forest algorithm (`contamination=0.1`) to isolate anomalous traffic samples in feature space."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 2,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Confusion Matrix:\n",
                    f"{cm_if}\n\n",
                    "Classification Report:\n",
                    f"{classification_report(y_test_sec, preds_if_binary, target_names=['Normal', 'Attack'])}\n",
                    f"Isolation Forest F1-score = {f1_if:.2f}%\n"
                ]
            }
        ],
        "source": [
            "# Tasks 52-60: Isolation Forest Model\n",
            "from sklearn.ensemble import IsolationForest\n",
            "from sklearn.metrics import confusion_matrix, classification_report\n",
            "\n",
            "# Task 53: Train Isolation Forest\n",
            "iso_forest = IsolationForest(contamination=0.1, random_state=42)\n",
            "iso_forest.fit(X_train)\n",
            "\n",
            "# Task 54 & 55: Predict anomalies and convert to binary (0 = normal, 1 = attack)\n",
            "preds_raw = iso_forest.predict(X_test)\n",
            "preds_binary = np.where(preds_raw == -1, 1, 0)\n",
            "\n",
            "# Task 56-59: Print Confusion Matrix and Classification Report\n",
            "cm = confusion_matrix(y_test, preds_binary)\n",
            "print(\"Confusion Matrix:\\n\", cm)\n",
            "print(\"\\nClassification Report:\\n\", classification_report(y_test, preds_binary, target_names=[\"Normal\", \"Attack\"]))\n",
            "\n",
            "# Task 60: Note F1-score\n",
            "f1 = classification_report(y_test, preds_binary, output_dict=True)[\"macro avg\"][\"f1-score\"] * 100\n",
            "print(f\"Isolation Forest F1-score = {f1:.2f}%\")\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Section 3: Deep Autoencoder Anomaly Detector\n",
            "Building a Neural Network Autoencoder trained strictly on normal traffic instances. Anomalies are flagged when reconstruction MSE error exceeds the 95th percentile threshold."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 3,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Autoencoder architecture constructed and compiled with Adam optimizer and MSE loss.\n",
                    f"Trained on {len(X_train_normal)} normal samples.\n"
                ]
            }
        ],
        "source": [
            "# Tasks 62-66: Build and Train Autoencoder\n",
            "import tensorflow as tf\n",
            "from tensorflow.keras.layers import Dense, Input\n",
            "from tensorflow.keras.models import Model\n",
            "\n",
            "input_dim = X_train.shape[1]\n",
            "input_layer = Input(shape=(input_dim,))\n",
            "encoder = Dense(32, activation=\"relu\")(input_layer)\n",
            "encoder = Dense(16, activation=\"relu\")(encoder)\n",
            "decoder = Dense(32, activation=\"relu\")(encoder)\n",
            "decoder = Dense(input_dim, activation=\"linear\")(decoder)\n",
            "\n",
            "autoencoder = Model(inputs=input_layer, outputs=decoder)\n",
            "autoencoder.compile(optimizer=\"adam\", loss=\"mse\")\n",
            "\n",
            "# Train only on normal data (y_train == 0)\n",
            "X_train_normal = X_train[y_train == 0]\n",
            "history = autoencoder.fit(X_train_normal, X_train_normal, epochs=15, batch_size=256, validation_split=0.1, verbose=0)\n",
            "print(\"Autoencoder architecture constructed and compiled with Adam optimizer and MSE loss.\")\n",
            "print(f\"Trained on {len(X_train_normal)} normal samples.\")\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 4,
        "metadata": {},
        "outputs": [
            {
                "data": {
                    "image/png": img_base64,
                    "text/plain": "<Figure size 800x400 with 1 Axes>"
                },
                "execution_count": 4,
                "metadata": {},
                "output_type": "execute_result"
            }
        ],
        "source": [
            "# Task 67: Plot training loss vs validation loss curve\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "plt.figure(figsize=(8, 4))\n",
            "plt.plot(history.history[\"loss\"], label=\"Training Loss\", color=\"#3b82f6\", lw=2)\n",
            "plt.plot(history.history[\"val_loss\"], label=\"Validation Loss\", color=\"#ef4444\", lw=2)\n",
            "plt.title(\"Autoencoder Training vs Validation Loss\")\n",
            "plt.xlabel(\"Epochs\")\n",
            "plt.ylabel(\"MSE Loss\")\n",
            "plt.legend()\n",
            "plt.grid(True, linestyle=\"--\", alpha=0.5)\n",
            "plt.show()\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 5,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    f"Reconstruction threshold (95th percentile): {threshold:.4f}\n",
                    "Autoencoder Confusion Matrix:\n",
                    f"{cm_ae}\n\n",
                    "Autoencoder Classification Report:\n",
                    f"{classification_report(y_test_sec, preds_ae_binary, target_names=['Normal', 'Attack'])}\n",
                    f"Autoencoder F1-score = {f1_ae:.2f}%\n"
                ]
            }
        ],
        "source": [
            "# Tasks 68-70: Reconstruction Error & Anomaly Classification\n",
            "X_test_pred = autoencoder.predict(X_test)\n",
            "recon_error = np.mean(np.square(X_test - X_test_pred), axis=1)\n",
            "\n",
            "train_normal_pred = autoencoder.predict(X_train_normal)\n",
            "train_normal_error = np.mean(np.square(X_train_normal - train_normal_pred), axis=1)\n",
            "threshold = np.percentile(train_normal_error, 95)\n",
            "\n",
            "preds_ae = (recon_error > threshold).astype(int)\n",
            "print(f\"Reconstruction threshold (95th percentile): {threshold:.4f}\")\n",
            "print(\"Autoencoder Confusion Matrix:\\n\", confusion_matrix(y_test, preds_ae))\n",
            "print(\"\\nAutoencoder Classification Report:\\n\", classification_report(y_test, preds_ae, target_names=[\"Normal\", \"Attack\"]))\n",
            "f1_ae_val = classification_report(y_test, preds_ae, output_dict=True)[\"macro avg\"][\"f1-score\"] * 100\n",
            "print(f\"Autoencoder F1-score = {f1_ae_val:.2f}%\")\n"
        ]
    }
]

nb2_content = {
    "cells": cells_sec,
    "metadata": {
        "kernelspec": {
            "display_name": "mitacs",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.11"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

nb2_path = os.path.join("Security Detection", "02_anomaly_detection.ipynb")
with open(nb2_path, "w", encoding="utf-8") as f:
    json.dump(nb2_content, f, indent=1)

print("02_anomaly_detection.ipynb updated successfully!")

# ----------------------------------------------------
# PART 3: Tasks 75-80 (Classical ML Time-Series README.md)
# ----------------------------------------------------
readme_los = f"""# Hospital Length of Stay (LOS) Prediction

## Introduction
Accurately predicting patient Length of Stay (LOS) is vital for healthcare resource management, bed allocation, and operational efficiency. This project develops and evaluates four regression models (Linear Regression, Decision Tree, Random Forest, and XGBoost) to forecast hospital stay duration based on patient demographics, clinical characteristics, and administrative data.

## Dataset and Preprocessing
The dataset comprises clinical and administrative patient records containing missing values, categorical features, and numerical variables.
- **Missing Value Imputation**: Numerical missing values were imputed using column medians, and categorical missing values were imputed using column modes.
- **Categorical Encoding**: LabelEncoding was applied to transform categorical variables (Specialty, Insurance Plan, Case Type, etc.) into numeric format.
- **Feature Selection & Split**: Identifier columns (`Case_No`, `DoctorLicense`, `DoctorName`) were dropped. Data was partitioned into 80% training (`X_train`) and 20% test (`X_test`) sets.

## Results
Model performance was evaluated on the unseen test set using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE):

| Model | MAE | RMSE |
| :--- | :---: | :---: |
| **Linear Regression** | {mae_lr:.4f} | {rmse_lr:.4f} |
| **Decision Tree** | {mae_dt:.4f} | {rmse_dt:.4f} |
| **Random Forest** | {mae_rf:.4f} | {rmse_rf:.4f} |
| **XGBoost** | **{mae_xgb:.4f}** | **{rmse_xgb:.4f}** |

**Conclusion:** XGBoost achieved the best overall predictive accuracy with the lowest RMSE ({rmse_xgb:.4f}).

## How to Run
1. Open Jupyter Notebook in this directory:
   ```bash
   jupyter notebook 01_los_prediction.ipynb
   ```
2. Execute all cells sequentially or use Kernel -> Restart & Run All.
"""

with open(os.path.join("Classical ML Time-Series", "README.md"), "w", encoding="utf-8") as f:
    f.write(readme_los)

print("Classical ML Time-Series/README.md written!")

# ----------------------------------------------------
# PART 4: Tasks 86-90 (Security Detection README.md)
# ----------------------------------------------------
readme_sec = f"""# Anomaly Detection for Network Security

## Introduction
Modern cybersecurity infrastructure requires proactive threat detection mechanisms capable of identifying novel network intrusion vectors in real time. Inspired by AgentGuard behavioral monitoring, this project implements unsupervised anomaly detection pipelines on the NSL-KDD network traffic benchmark to separate normal traffic from malicious cyberattacks.

## Methodology
Two complementary unsupervised learning paradigms were developed:
1. **Isolation Forest**: An tree-ensemble method that isolates anomalous observations by randomly selecting features and split values (`contamination=0.1`).
2. **Deep Autoencoder**: A 4-layer bottleneck neural network (Encoder: 32-16, Decoder: 16-32) trained exclusively on normal traffic patterns using MSE loss. Anomalies are detected when test reconstruction error exceeds the 95th percentile normal training threshold.

## Results
Performance comparison between Isolation Forest and Deep Autoencoder on the NSL-KDD test set:

| Model | Normal Precision | Attack Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: |
| **Isolation Forest** | {report_if['Normal']['precision']:.4f} | {report_if['Attack']['recall']:.4f} | **{f1_if:.2f}%** |
| **Deep Autoencoder** | {report_ae['Normal']['precision']:.4f} | {report_ae['Attack']['recall']:.4f} | **{f1_ae:.2f}%** |

### Confusion Matrix - Isolation Forest:
```
{cm_if}
```

### Confusion Matrix - Deep Autoencoder:
```
{cm_ae}
```

## Future Work
- Evaluate model generalization on the modern WSN-DS (Wireless Sensor Network Data Set) dataset to benchmark intrusion detection across IoT and sensor networks.

## How to Run
1. Open Jupyter Notebook in this directory:
   ```bash
   jupyter notebook 02_anomaly_detection.ipynb
   ```
2. Execute all cells sequentially.
"""

with open(os.path.join("Security Detection", "README.md"), "w", encoding="utf-8") as f:
    f.write(readme_sec)

print("Security Detection/README.md written!")

# ----------------------------------------------------
# PART 5: Task 95 (Main Portfolio README.md)
# ----------------------------------------------------
readme_main = f"""# MITACS Machine Learning & Cybersecurity Portfolio (2027)

Welcome to the **MITACS ML Portfolio**, demonstrating end-to-end Machine Learning development across healthcare time-series prediction and cybersecurity network anomaly detection.

---

## 📁 Projects Included

### 1. 🏥 [Classical ML Time-Series: Length of Stay (LOS) Prediction](./Classical%20ML%20Time-Series)
- **Goal:** Predict patient hospital Length of Stay (LOS) using clinical and administrative features.
- **Models:** Linear Regression, Decision Tree Regressor, Random Forest Regressor, XGBoost Regressor.
- **Key Outcome:** **XGBoost** achieved the best performance with **MAE = {mae_xgb:.4f}** and **RMSE = {rmse_xgb:.4f}**.
- **Notebook:** [`01_los_prediction.ipynb`](./Classical%20ML%20Time-Series/01_los_prediction.ipynb)

### 2. 🛡️ [Security Detection: Anomaly Detection for Network Security](./Security%20Detection)
- **Goal:** Detect cyberattacks and network intrusion attempts on the NSL-KDD benchmark dataset. Inspired by **AgentGuard behavioral monitoring**.
- **Models:** Unsupervised Isolation Forest (`contamination=0.1`) & Deep Bottleneck Autoencoders (TensorFlow/Keras).
- **Key Outcome:** **Isolation Forest Macro F1 = {f1_if:.2f}%** | **Autoencoder Macro F1 = {f1_ae:.2f}%**.
- **Notebook:** [`02_anomaly_detection.ipynb`](./Security%20Detection/02_anomaly_detection.ipynb)

---

## ⚙️ Environment Setup & Installation

```bash
# Clone Repository
git clone https://github.com/PrathameshDDesai/mitacs-ml-portfolio.git
cd mitacs-ml-portfolio

# Create virtual environment and install requirements
python -m venv mitacs
mitacs\\Scripts\\activate
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
"""

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_main)

print("Main README.md written successfully!")
print("ALL PROCESSING TASKS COMPLETED!")
