import os
os.environ['KERAS_BACKEND'] = 'torch'

import torch
# EXTREME MEMORY SAVING FOR RENDER FREE TIER:
torch.set_grad_enabled(False)
torch.set_num_threads(1)

import keras
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import gc

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
        # Force garbage collection before prediction to clear RAM
        gc.collect()
        
        img_array = prepare_image(file.read())
        
        # We use a raw tensor forward pass instead of model.predict() to save massive amounts of RAM
        # model.predict() creates data loaders and callbacks which cause OOM on 512MB RAM
        img_tensor = torch.tensor(img_array)
        final_preds = mobilenet_model(img_tensor, training=False).numpy()
            
        class_idx = np.argmax(final_preds[0])
        class_name = CLASSES[class_idx]
        confidence = float(final_preds[0][class_idx]) * 100
        probabilities = {CLASSES[i]: float(final_preds[0][i]) * 100 for i in range(len(CLASSES))}
        
        return jsonify({'prediction': class_name, 'confidence': confidence, 'probabilities': probabilities, 'model_used': 'mobilenet_v2'})
    except Exception as e:
        return jsonify({'error': f"Inference crashed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
