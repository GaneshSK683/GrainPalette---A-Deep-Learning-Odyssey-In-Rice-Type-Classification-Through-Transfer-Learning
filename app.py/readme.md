from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(_name_)

model = tf.keras.models.load_model('training/rice_model.h5')
class_names = ['Basmati', 'Jasmine', 'Arborio', 'Brown', 'White']

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part in request", 400

    file = request.files['file']

    if file.filename == '':
        return "No file selected", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
img = image.load_img(filepath, target_size=(150, 150))  # match model input shape
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # shape = (1, 150, 150, 3)

predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]
    return render_template('index.html', prediction=predicted_class, image_path=filepath)

if _name_ == '_main_':
    app.run(debug=True)
