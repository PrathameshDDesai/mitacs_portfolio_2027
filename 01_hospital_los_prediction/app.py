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

print("[Healthcare Clinical OS] Loading Length of Stay (LOS) predictive models...")
rf_los = joblib.load(os.path.join(SAVED_MODELS_DIR, "rf_model.joblib"))
xgb_los = joblib.load(os.path.join(SAVED_MODELS_DIR, "xgb_model.joblib"))
los_encoders = joblib.load(os.path.join(SAVED_MODELS_DIR, "encoders.joblib"))
los_feature_names = joblib.load(os.path.join(SAVED_MODELS_DIR, "feature_names.joblib"))

print("[Healthcare Clinical OS] Models loaded successfully!")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "project": "01_hospital_los_prediction", "port": int(os.environ.get("PORT", 5001))})

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
        variance = abs(rf_pred - xgb_pred)

        cmi_val = float(data.get("CMI Value", 1.5))
        severity = str(data.get("Severity", "Medium"))
        specialty = str(data.get("Specialty", "General Medicine"))

        # Determine Risk Level & Clinical Bed Allocation Guidance
        clinical_notes = []
        if ensemble_pred >= 6.0:
            risk_level = "EXTENDED"
            risk_category = "Extended Stay (> 6 Days)"
            bed_guidance = "High-Dependency Unit / ICU Bed Allocation Required"
            clinical_notes.append("Multi-specialty daily rounds recommended for prolonged stay management.")
            clinical_notes.append("High Case Mix Index (CMI) indicates complex resource utilization.")
        elif ensemble_pred >= 3.5:
            risk_level = "MODERATE"
            risk_category = "Moderate Stay (3-6 Days)"
            bed_guidance = "Specialized Inpatient Ward Bed Allocation"
            clinical_notes.append("Schedule step-down telemetry evaluation on Day 3.")
            clinical_notes.append("Coordinate discharge planning with primary attending physician.")
        else:
            risk_level = "LOW"
            risk_category = "Short Stay (< 3 Days)"
            bed_guidance = "Standard Acute Care Ward Bed"
            clinical_notes.append("Routine short-stay protocol. Target discharge window within 48-72 hours.")
            clinical_notes.append("Low complexity case profile; standard post-admission monitoring.")

        if severity in ["High", "Extreme"]:
            clinical_notes.append(f"Elevated admission severity ({severity}) flagged for priority nursing observation.")

        return jsonify({
            "status": "success",
            "rf_prediction": round(rf_pred, 2),
            "xgb_prediction": round(xgb_pred, 2),
            "ensemble_prediction": round(ensemble_pred, 2),
            "model_variance": round(variance, 2),
            "risk_level": risk_level,
            "risk_category": risk_category,
            "bed_guidance": bed_guidance,
            "clinical_notes": clinical_notes,
            "cmi_value": cmi_val,
            "severity": severity,
            "specialty": specialty
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print("================================================================")
    print(f"Healthcare Clinical OS — Hospital LOS Predictor Server")
    print(f"Production Waitress WSGI Server listening on http://localhost:{port}")
    print("================================================================")
    serve(app, host="0.0.0.0", port=port, threads=8)
