from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageDraw, ImageFont
import io
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import base64
import numpy as np
import cv2

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO('best.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read image file
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Perform inference
        results = model(image)
        
        # Draw bounding boxes on the image
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            
            for box, cls, conf in zip(boxes, classes, confs):
                x1, y1, x2, y2 = box
                label = f"{model.names[cls]}: {conf:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert the annotated image to base64
        _, buffer = cv2.imencode('.png', image)
        img_str = base64.b64encode(buffer).decode()

        return jsonify({'image': img_str})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)