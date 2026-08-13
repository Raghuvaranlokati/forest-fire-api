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

from keras.applications.vgg16 import preprocess_input as vgg_preprocess
from keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

vgg16_model = None
mobilenet_model = None
vgg_error = "Disabled to save memory on Render Free Tier (OOM prevention)"
mobilenet_error = None
CLASSES = ['fire', 'nofire', 'smoke', 'smokefire']
IMG_SIZE = (224, 224)

print("Loading models for Render API...")
# We ONLY load MobileNetV2 because VGG16 + TensorFlow exceeds 512MB RAM on free tier!
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
        "status": "API is running. Use POST /predict to classify images.",
        "vgg16_loaded": vgg16_model is not None,
        "mobilenet_loaded": mobilenet_model is not None,
        "vgg_error": vgg_error,
        "mobilenet_error": mobilenet_error
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    model_choice = request.form.get('model', 'ensemble') 
    
    try:
        img_array = prepare_image(file.read())
        preds_vgg = None
        preds_mobilenet = None
        
        if model_choice in ['vgg16', 'ensemble'] and vgg16_model is not None:
            preds_vgg = vgg16_model.predict(img_array, verbose=0)
            
        if model_choice in ['mobilenet', 'ensemble'] and mobilenet_model is not None:
            preds_mobilenet = mobilenet_model.predict(img_array, verbose=0)
            
        # Fallback logic to prevent NoneType errors
        if model_choice == 'ensemble':
            if preds_vgg is not None and preds_mobilenet is not None:
                final_preds = (preds_vgg + preds_mobilenet) / 2.0
            elif preds_mobilenet is not None:
                final_preds = preds_mobilenet # Fallback to mobilenet if VGG failed/OOM
            elif preds_vgg is not None:
                final_preds = preds_vgg
            else:
                return jsonify({'error': 'No models could be loaded. Prediction failed.'}), 500
        elif model_choice == 'vgg16':
            if preds_vgg is None:
                return jsonify({'error': 'VGG16 model is disabled due to memory limits.'}), 400
            final_preds = preds_vgg
        elif model_choice == 'mobilenet':
            if preds_mobilenet is None:
                return jsonify({'error': 'MobileNet model failed to load.'}), 500
            final_preds = preds_mobilenet
        else:
            return jsonify({'error': 'Invalid model choice'}), 400
            
        class_idx = np.argmax(final_preds[0])
        class_name = CLASSES[class_idx]
        confidence = float(final_preds[0][class_idx]) * 100
        probabilities = {CLASSES[i]: float(final_preds[0][i]) * 100 for i in range(len(CLASSES))}
        
        return jsonify({'prediction': class_name, 'confidence': confidence, 'probabilities': probabilities, 'model_used': model_choice})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
