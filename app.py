import os
os.environ['KERAS_BACKEND'] = 'torch'

import torch
import keras
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

app = Flask(__name__)
# Allow CORS for all domains so your React app on Vercel can access it
CORS(app, resources={r"/*": {"origins": "*"}})

from keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

mobilenet_model = None
mobilenet_error = None
CLASSES = ['fire', 'nofire', 'smoke', 'smokefire']
IMG_SIZE = (224, 224)

print("Loading MobileNetV2 for Render API...")
try:
    mobilenet_model = keras.models.load_model('mobilenet_fire_model(73.00%).keras', custom_objects={'preprocess_input': mobilenet_preprocess})
    print("MobileNetV2 loaded successfully.")
except Exception as e:
    mobilenet_error = str(e)
    print(f"Error loading MobileNetV2: {e}")

def prepare_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "API is running. Only MobileNetV2 is loaded to stay within the 512MB RAM limit.",
        "mobilenet_loaded": mobilenet_model is not None,
        "mobilenet_error": mobilenet_error
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if mobilenet_model is None:
        return jsonify({'error': f'MobileNet model failed to load during startup: {mobilenet_error}'}), 500
        
    try:
        img_array = prepare_image(file.read())
        
        # We completely ignore the 'model' parameter from the frontend and ONLY use MobileNet
        final_preds = mobilenet_model.predict(img_array, verbose=0)
            
        class_idx = np.argmax(final_preds[0])
        class_name = CLASSES[class_idx]
        confidence = float(final_preds[0][class_idx]) * 100
        probabilities = {CLASSES[i]: float(final_preds[0][i]) * 100 for i in range(len(CLASSES))}
        
        return jsonify({'prediction': class_name, 'confidence': confidence, 'probabilities': probabilities, 'model_used': 'mobilenet_v2'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
