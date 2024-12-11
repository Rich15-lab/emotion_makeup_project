from flask import Flask, request, jsonify
import cv2
import numpy as np
from deepface import DeepFace

app = Flask(__name__)

# Emotion to Makeup Recommendations
product_recommendations = {
    'happy': ['Bold Red Lipstick - Maybelline SuperStay Matte Ink', 
              'Golden Glow Blush - Fenty Beauty'],
    'sad': ['Subtle Nude Lipstick - Revlon', 
            'Matte Beige Eyeshadow - NYX Professional'],
    'angry': ['Fiery Orange Lipstick - MAC Cosmetics', 
              'Smoky Black Eyeliner - Urban Decay'],
    'surprise': ['Shimmery Pink Lip Gloss - Glossier', 
                 'Peach Highlighter - Tarte'],
    'neutral': ['Natural Brown Lipstick - Bobbi Brown', 
                'Soft Pink Blush - Rare Beauty'],
    'fear': ['Deep Purple Lipstick - NARS', 
             'Violet Eyeshadow - Anastasia Beverly Hills']
}

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Receive and process the image
        image_file = request.files['image']
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 
cv2.IMREAD_COLOR)

        # Resize to save memory
        image = cv2.resize(image, (224, 224))

        # Analyze the emotion
        analysis = DeepFace.analyze(image, actions=['emotion'], 
models={'emotion': 'fer'})
        emotion = analysis['dominant_emotion']
        recommendations = product_recommendations.get(emotion, [])

        return jsonify({
            'emotion': emotion,
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

