from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)

# Load the trained model
model = load_model('training/rice_model.h5')

# Your class names — make sure these match your model's output
class_names = ['Basmati', 'Jasmine', 'Arborio', 'Brown', 'White']

# Configure upload folder
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

    # Save with a unique filename to avoid permission issues
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    # Preprocess image to match model's input shape (150x150x3)
    img = image.load_img(filepath, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 150, 150, 3)

    # Predict using the model
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    return render_template('index.html', prediction=predicted_class, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
