import tensorflow as tf
import cv2
from flask import Flask, render_template, redirect, request, url_for
import os
import numpy as np

app = Flask(__name__)
CLASS_NAMES = {
    '0': 'Speed Limit: 20',
    '1': 'Speed Limit: 30',
    '2': 'Speed Limit: 50',
    '3': 'Speed Limit: 60',
    '4': 'Speed Limit: 70',
    '5': 'Speed Limit: 80',
    '6': 'Speed Limit: 80',
    '7': 'Speed Limit: 100',
    '8': 'Speed Limit: 120',
    '9': 'No Overtaking',
    '10': 'No Overtaking: Trucks',
    '11': 'Right of way: Crossing',
    '12': 'Right of way: General',
    '13': 'Give way',
    '14': 'Stop',
    '15': 'No way: General',
    '16': 'No way: Trucks',
    '17': 'No way: One way',
    '18': 'Attention',
    '19': 'Attention: Left turn',
    '20': 'Attention: Right turn',
    '21': 'Attention: Curvy',
    '22': 'Attention: Bumpers',
    '23': 'Attention: Slippery',
    '24': 'Attention: Bottleneck',
    '25': 'Attention: Construction',
    '26': 'Attention: Traffic Light',
    '27': 'Attention: Pedistrians',
    '28': 'Attention: Children',
    '29': 'Attention: Bikes',
    '30': 'Attention: Snow',
    '31': 'Attention: Deer',
    '32': 'Lifted',
    '33': 'Turn right',
    '34': 'Turn left',
    '35': 'Go straight',
    '36': 'Go straight or turn right',
    '37': 'Go straight or turn left',
    '38': 'Turn down right',
    '39': 'Turn down left',
    '40': 'Roundabout',
    '41': 'No overtaking',
    '42': 'No overtaking trucks'
}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def predict():
    # Load model, save uploaded image to static directory
    model = tf.keras.models.load_model('trained_model.keras')
    image_file = request.files['file']
    path = os.path.join(os.getcwd(), "static", image_file.filename)
    image_file.save(path)

    # Preprocess image, get its predicted class and output it.
    img = preprocess_image(path)
    predictions = model.predict(np.array([img]))
    predicted_class_index = np.argmax(predictions[0]) # Gets the largest class probability
    class_prediction = CLASS_NAMES[str(int(predicted_class_index))]
    image_url = url_for('static', filename=image_file.filename)

    return render_template("index.html", prediction=class_prediction, image = image_url)


def preprocess_image(path):
    img = cv2.imread(path)
    resized_image = cv2.resize(img, (30, 30))
    return resized_image

if __name__ == '__main__':
    app.run(debug=True)