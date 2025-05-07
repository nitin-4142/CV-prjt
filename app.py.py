from flask import Flask, render_template, request
import os
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import json

app = Flask(__name__)

# Load model
MODEL_PATH = 'C:\CV_PROJECT\model\dt_model.pkl'
model = load_model(MODEL_PATH)

# FER-2013 emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Utility: Get all image paths from subfolders
def get_all_image_paths(root_folder='static/images'):
    image_files = []
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                relative_path = os.path.relpath(os.path.join(subdir, file), start='static')
                image_files.append(relative_path.replace("\\", "/"))  # For Windows compatibility
    return image_files

@app.route('/')
def index():
    image_files = get_all_image_paths()
    return render_template('index.html', images=image_files)

@app.route('/predict', methods=['POST'])
def predict():
    selected_image = request.form['selected_image']
    img_path = os.path.join('static', selected_image)

    # Preprocess image
    img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict emotion
    prediction = model.predict(img_array)
    emotion = emotion_labels[np.argmax(prediction)]

    return render_template('final_result.html',
                           selected_image=selected_image,
                           emotion=emotion)

@app.route('/metrics')
def view_metrics():
    with open('metrics.json') as f:
        data = json.load(f)

    return render_template('result.html',
                           accuracy=data['accuracy'],
                           avg_iou=data['avg_iou'],
                           confusion_matrix=data['confusion_matrix'],
                           filename=data.get('example_image', 'default.png'))

if __name__ == '__main__':
    app.run(debug=True)
