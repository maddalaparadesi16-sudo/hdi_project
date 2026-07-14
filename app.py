import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

model_data = None

def load_model():
    global model_data
    model_path = os.path.join(os.path.dirname(__file__), "models", "hdi_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
            print("Model loaded successfully.")
    else:
        print("Warning: Model file not found. Please run train.py first.")

load_model()

def get_hdi_tier(score):
    if score >= 0.800: return "Very High", "emerald"
    elif score >= 0.700: return "High", "teal"
    elif score >= 0.550: return "Medium", "amber"
    else: return "Low", "rose"

@app.route("/")
def home():
    if model_data is None:
        load_model()
    metrics = {}
    if model_data is not None:
        metrics = model_data.get("metrics", {})
    return render_template("index.html", r2=metrics.get("r2", 0.0))

@app.route("/predict", methods=["POST"])
def predict():
    global model_data
    if model_data is None:
        load_model()
        if model_data is None:
            return jsonify({"error": "Model is not loaded. Run training script first."}), 500
            
    try:
        data = request.json if request.is_json else request.form
            
        life_exp = float(data.get("life_expectancy", 0))
        expected_schooling = float(data.get("expected_schooling", 0))
        mean_schooling = float(data.get("mean_schooling", 0))
        gni_pc = float(data.get("gni_per_capita", 0))
        
        if not (20 <= life_exp <= 100):
            return jsonify({"error": "Life expectancy must be between 20 and 100 years."}), 400
        if not (0 <= expected_schooling <= 30):
            return jsonify({"error": "Expected schooling must be between 0 and 30 years."}), 400
        if not (0 <= mean_schooling <= 25):
            return jsonify({"error": "Mean schooling must be between 0 and 25 years."}), 400
        if not (100 <= gni_pc <= 200000):
            return jsonify({"error": "GNI per capita must be between $100 and $200,000."}), 400
            
        # Passing as DataFrame fixes sklearn feature-name warnings
        feature_names = model_data.get("features", ["Life Expectancy", "Expected Years of Schooling", "Mean Years of Schooling", "GNI per Capita"])
        features_df = pd.DataFrame([[life_exp, expected_schooling, mean_schooling, gni_pc]], columns=feature_names)
        
        model = model_data["model"]
        prediction = model.predict(features_df)[0]
        
        predicted_hdi = min(max(float(prediction), 0.0), 1.0)
        predicted_hdi = round(predicted_hdi, 3)
        tier, color = get_hdi_tier(predicted_hdi)
        
        lei = min(max((life_exp - 20) / (85 - 20), 0.0), 1.0)
        eysi = min(max(expected_schooling / 18, 0.0), 1.0)
        mysi = min(max(mean_schooling / 15, 0.0), 1.0)
        ei = (eysi + mysi) / 2
        gni_capped = min(max(gni_pc, 100), 75000)
        ii = (np.log(gni_capped) - np.log(100)) / (np.log(75000) - np.log(100))
        
        return jsonify({
            "predicted_hdi": predicted_hdi,
            "tier": tier,
            "color": color,
            "dimensions": {
                "health_index": round(lei, 3),
                "education_index": round(ei, 3),
                "income_index": round(ii, 3)
            }
        })
        
    except ValueError:
        return jsonify({"error": "Invalid numerical values provided."}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)