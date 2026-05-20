import os
# Suppress initialization logging flags
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf


app = Flask(__name__)


IMG_SIZE = 128


# Label mapping based on alphabetical dataset folders
CLASS_LABELS = {
    0: "keyboard",
    1: "mouse"
}


# Rebuild the exact model architecture from training
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),


        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),


        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),


        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),


        tf.keras.layers.Flatten(),


        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),


        tf.keras.layers.Dense(2, activation='softmax')  # Matches keyboard and mouse
    ])
    return model


# Initialize and load weights safely
weights_path = "device_model.weights.h5"


try:
    if os.path.exists(weights_path):
        model = create_model()
        model.load_weights(weights_path)
        print("Successfully loaded model architecture and weights!")
    else:
        model = None
        print(f"CRITICAL: Weights file '{weights_path}' not found in the application directory!")
except Exception as e:
    model = None
    print(f"CRITICAL: Failed to load model weights. Error: {e}")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model is not initialized. Check server terminal logs."}), 500
       
    try:
        # Check if file exists in request payload keys
        if 'file' not in request.files:
            return jsonify({
                "error": "No file part in the request. Your form field or key name MUST be exactly 'file'."
            }), 400
           
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400


        # Read and preprocess the incoming image
        img_bytes = file.read()
       
        # channels (RGB) to guarantee compatibility with Input shape
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img_array = tf.expand_dims(img, 0) / 255.0  # Normalize rescaling matching training


        # Perform prediction safely
        predictions = model.predict(img_array)
       
        # Extract indexes and compute metrics
        predicted_class_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
       
        # Translate the numerical index to the actual string name
        predicted_label = CLASS_LABELS.get(predicted_class_index, "unknown")


        return jsonify({
            "class_index": predicted_class_index,
            "class_label": predicted_label,
            "confidence": confidence,
            "all_predictions": predictions.tolist()
        })


    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)

    # Save model locally
joblib.dump(model, 'iris_model.pkl')
print("Model trained and saved successfully!")