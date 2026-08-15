import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from waitress import serve

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

print("[Project 1: Hospital LOS] Loading models...")
rf_los = joblib.load(os.path.join(SAVED_MODELS_DIR, "rf_model.joblib"))
xgb_los = joblib.load(os.path.join(SAVED_MODELS_DIR, "xgb_model.joblib"))
los_encoders = joblib.load(os.path.join(SAVED_MODELS_DIR, "encoders.joblib"))
los_feature_names = joblib.load(os.path.join(SAVED_MODELS_DIR, "feature_names.joblib"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "project": "01_hospital_los_prediction", "port": 5001})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json or {}
        row = {}
        for feature in los_feature_names:
            val = data.get(feature, 0)
            if feature in los_encoders:
                le = los_encoders[feature]
                str_val = str(val)
                if str_val in le.classes_:
                    encoded = le.transform([str_val])[0]
                else:
                    encoded = 0
                row[feature] = float(encoded)
            else:
                try:
                    row[feature] = float(val)
                except (ValueError, TypeError):
                    row[feature] = 0.0

        input_df = pd.DataFrame([row], columns=los_feature_names)
        
        rf_pred = float(rf_los.predict(input_df)[0])
        xgb_pred = float(xgb_los.predict(input_df)[0])
        ensemble_pred = (rf_pred + xgb_pred) / 2.0

        risk_category = "Low Stay (< 3 Days)"
        if ensemble_pred >= 6.0:
            risk_category = "Extended Stay (> 6 Days)"
        elif ensemble_pred >= 3.5:
            risk_category = "Moderate Stay (3-6 Days)"

        return jsonify({
            "status": "success",
            "rf_prediction": round(rf_pred, 2),
            "xgb_prediction": round(xgb_pred, 2),
            "ensemble_prediction": round(ensemble_pred, 2),
            "risk_category": risk_category
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    print("================================================================")
    print("Project 1: Hospital Length of Stay (LOS) Predictor Server")
    print("Production Waitress WSGI Server listening on http://localhost:5001")
    print("================================================================")
    serve(app, host="0.0.0.0", port=5001, threads=8)
