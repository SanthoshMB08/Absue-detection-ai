from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from flask_cors import CORS  # ✅ Import Flask-CORS

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all routes

# Load the trained model and tokenizer
MODEL_PATH = "./abuse_detection_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Set model to evaluation mode
model.eval()

# Function to predict comment category
def predict_comment(comment):
    inputs = tokenizer(comment, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()
    return "Non-Abusive ✅" if predicted_label == 1 else "Abusive ❌"

# Home Route
@app.route("/")
def home():
    return render_template("ui.html")

# API Route to get predictions
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    comment = data.get("comment", "")
    if not comment:
        return jsonify({"error": "Empty comment"}), 400
    prediction = predict_comment(comment)
    return jsonify({"result": prediction})

if __name__ == "__main__":
    app.run(debug=True)
