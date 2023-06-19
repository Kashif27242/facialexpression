from PIL import Image
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2


app = Flask(__name__)
model = load_model("best_model.h5")
face_haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def predict_emotion(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_pixels = np.expand_dims(img_array, axis=0)
    img_pixels = img_pixels / 255.0

    predictions = model.predict(img_pixels)
    max_index = np.argmax(predictions[0])
    emotions = ['angry', 'disgust', 'fear',
                'happy', 'sad', 'surprise', 'neutral']
    predicted_emotion = emotions[max_index]

    return predicted_emotion


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_path = 'uploads/' + image_file.filename
            image_file.save(image_path)
            predicted_emotion = predict_emotion(image_path)
            return render_template('result.html', image_path=image_path, emotion=predicted_emotion)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
