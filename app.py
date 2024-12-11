from flask import Flask, request, jsonify
import cv2
import numpy as np
from deepface import DeepFace
import tensorflow as tf

# Optimize TensorFlow memory usage
physical_devices = tf.config.list_physical_devices('CPU')
if physical_devices:
    tf.config.experimental.set_virtual_device_configuration(
        physical_devices[0],
        
[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=256)]
    )

app = Flask(__name__)

# Lightweight emotion model
product_recommendations = {
    'happy': ['Bold Red Lipstick - Maybelline SuperStay Matte Ink', 
'Golden Glow Blush - Fenty Beauty'],
Eyeshadow - NYX Professional'],
 'sad': ['Subtle Nude Lipstick - Revlon', 'Matte Beige Eyeshadow - NYX 
Professional'],
    'angry': ['Fiery Orange Lipstick - MAC Cosmetics', 'Smoky Black 
Eyeliner - Urban Decay'],
    'surprise': ['Shimmery Pink Lip Gloss - Glossier', 'Peach Highlighter 
- Tarte'],
    'neutral': ['Natural Brown Lipstick - Bobbi Brown', 'Soft Pink Blush - 
Rare Beauty'],
    'fear': ['Deep Purple Lipstick - NARS', 'Violet Eyeshadow - Anastasia 
Beverly Hills']
}

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        image_file = request.files['image']
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 
cv2.IMREAD_COLOR)
        analysis = DeepFace.analyze(image, actions=['emotion'], 
models={'emotion': 'fer'})

        emotion = analysis[0]['dominant_emotion'] if isinstance(analysis, 
list) else analysis['dominant_emotion']
        recommendations = product_recommendations.get(emotion, [])

        return jsonify({
            'emotion': emotion,
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

