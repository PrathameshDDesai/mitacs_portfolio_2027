import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

print("[01_hospital_los_prediction] Training and exporting models...")

df_los = pd.read_csv(os.path.join(BASE_DIR, "dummy_hospital_generated.csv"))

# Impute missing values
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

# Train Models
rf_los = RandomForestRegressor(n_estimators=100, random_state=42)
rf_los.fit(X_los, y_los)

xgb_los = XGBRegressor(n_estimators=100, random_state=42)
xgb_los.fit(X_los, y_los)

# Save artifacts inside project 1 folder
joblib.dump(rf_los, os.path.join(SAVED_MODELS_DIR, "rf_model.joblib"))
joblib.dump(xgb_los, os.path.join(SAVED_MODELS_DIR, "xgb_model.joblib"))
joblib.dump(encoders, os.path.join(SAVED_MODELS_DIR, "encoders.joblib"))
joblib.dump(list(X_los.columns), os.path.join(SAVED_MODELS_DIR, "feature_names.joblib"))

print("[01_hospital_los_prediction] Models exported successfully to saved_models/!")
