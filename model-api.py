from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from difflib import get_close_matches
import os

# Flask setup
app = Flask(__name__)
CORS(app)

# Load dataset
df = pd.read_csv('120_copy_dataset.csv')

# Prepare encoders and scaler
labels = df["diseases"]
label_encoder = LabelEncoder()
label_encoder.fit(labels)

features = df.drop("diseases", axis=1)
scaler = StandardScaler()
scaler.fit(features)

# Load model
model = tf.keras.models.load_model("89% accuracy 120 copy.keras")  # 👈 change name if different

# Symptom list
symptom_list = features.columns.tolist()

# Suggest symptoms based on input
def suggest_symptoms(typed_symptom):
    return get_close_matches(typed_symptom, symptom_list, n=10, cutoff=0.5)

# Routes

@app.route('/')
def home():
    return "Disease Prediction API is running."

@app.route('/suggest', methods=['POST'])
def suggest():
    data = request.get_json()
    symptom = data.get("symptom", "").lower()
    suggestions = suggest_symptoms(symptom)
    return jsonify({"suggestions": suggestions})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data.get("symptoms", [])

    if not symptoms:
        return jsonify({"error": "No symptoms provided."}), 400

    # Process symptom vector
    input_vector = np.zeros(len(features.columns))
    for s in symptoms:
        s = s.lower().strip()
        if s in features.columns:
            input_vector[features.columns.get_loc(s)] = 1

    if not np.any(input_vector):
        return jsonify({"error": "No valid symptoms matched."}), 400

    input_scaled = scaler.transform([input_vector])
    prediction = model.predict(input_scaled)

    top_5_indices = np.argsort(prediction[0])[::-1][:5]
    top_5_probs = prediction[0][top_5_indices]
    top_5_classes = label_encoder.inverse_transform(top_5_indices)

    result = [
        {"disease": top_5_classes[i], "probability": float(top_5_probs[i])}
        for i in range(5)
    ]
    return jsonify({"predictions": result})

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port, debug=True)
