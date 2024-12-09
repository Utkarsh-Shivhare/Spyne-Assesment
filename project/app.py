# app.py
from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import io
import torch.nn as nn
import numpy as np

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_efficientnet_model(num_classes):
    """
    Returns a pre-trained EfficientNet model with a modified classifier head for classification.
    """
    model = efficientnet_b0(pretrained=True)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    return model.to(device)

class_names = ['0', '40', '90', '130', '180', '230', '270','320']
model = get_efficientnet_model(num_classes=len(class_names))
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Define preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        # Read and preprocess image
        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)  # Output shape: [1, 8]
            predicted_class = torch.argmax(output, dim=1).item()  # Get the class index
            confidence = torch.softmax(output, dim=1).max().item() 
                
            # Calculate confidence score (example implementation)
            # confidence = np.random.uniform(0.7, 0.99)  # Replace with actual confidence calculation

        aclass_names = ['0', '130', '180', '230', '270', '320', '40', '90']
        angle = aclass_names[predicted_class]

        return jsonify({"angle": angle, "confidence": confidence})

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)