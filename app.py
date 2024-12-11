from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# Lightweight emotion-makeup recommendations
product_recommendations = {
    'happy': ['Bold Red Lipstick - Maybelline', 'Golden Glow Blush - Fenty 
Beauty'],
    'sad': ['Subtle Nude Lipstick - Revlon', 'Matte Beige Eyeshadow - 
NYX'],
    'angry': ['Fiery Orange Lipstick - MAC', 'Smoky Black Eyeliner - Urban 
Decay'],
    'surprise': ['Shimmery Pink Lip Gloss - Glossier', 'Peach Highlighter 
- Tarte'],
    'neutral': ['Natural Brown Lipstick - Bobbi Brown', 'Soft Pink Blush - 
Rare Beauty'],
    'fear': ['Deep Purple Lipstick - NARS', 'Violet Eyeshadow - 
Anastasia']
}

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Receive and resize the image
        image_file = request.files['image']
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 
cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))

        # Fake emotion detection for demo purposes
        emotion = 'neutral'  # Replace with actual logic or models
        recommendations = product_recommendations.get(emotion, [])

        return jsonify({
            'emotion': emotion,
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

