from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

# Load the TorchScript model
try:
    model = torch.jit.load("model_scripted.pt")
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class labels
class_names = ["defect", "longberry", "peaberry", "premium"]

# Initialize Flask app
app = Flask(__name__)

# Home Route
@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Coffee Bean Quality Estimation API!"})

# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model failed to load"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = transform(img).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            prediction = class_names[predicted.item()]

        return jsonify({"prediction": prediction})
    
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

# Run Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
