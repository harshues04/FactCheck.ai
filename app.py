from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model
model = load_model('model/misinfo_classify.keras')

# Constants for text preprocessing
VOCAB_SIZE = 20000
TEXT_SEQUENCE_LENGTH = 100
tokenizer = Tokenizer(num_words=VOCAB_SIZE)

# Constants for image preprocessing
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Define upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to preprocess text input
def preprocess_text(text):
    # Tokenize and pad the text input
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=TEXT_SEQUENCE_LENGTH)
    return padded_sequence

# Function to preprocess image input
def preprocess_image(image_path):
    # Load the image, resize it, and normalize pixel values
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/classify_text', methods=['POST'])
def classify_text_route():
    data = request.json
    text = data.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # Preprocess and predict
        preprocessed_text = preprocess_text(text)
        prediction = model.predict([np.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3)), preprocessed_text])
        label = "real" if prediction[0][0] < 0.5 else "fake"
        return jsonify({"prediction": label})
    except Exception as e:
        return jsonify({"error": f"Error processing text: {str(e)}"}), 500

@app.route('/classify_file', methods=['POST'])
def classify_file_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Save file and preprocess
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        preprocessed_image = preprocess_image(file_path)
        
        # Predict
        prediction = model.predict([preprocessed_image, np.zeros((1, TEXT_SEQUENCE_LENGTH))])
        label = "real" if prediction[0][0] < 0.5 else "fake"
        return jsonify({"prediction": label})
    except Exception as e:
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)