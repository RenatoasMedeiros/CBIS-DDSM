from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import base64
import io
import tensorflow as tf

# --- Setup ---
app = Flask(__name__)
DETECTION_THRESHOLD = 1
# Define the standard input size for your models.
MODEL_INPUT_SIZE = (224, 224)

# Load models
# It's good practice to wrap model loading in a try-except block
try:
    model_classification = load_model('model_classification.keras')
    model_detection = load_model('model_detection.keras')
except Exception as e:
    print(f"Error loading models: {e}")
    # Handle the error appropriately, maybe exit or use dummy models
    # For this example, we'll let it raise the error.


# --- Utility Functions ---

def preprocess_image(pil_img, size):
    """
    Preprocesses a PIL image for a Keras model.
    Resizes to the target size, converts to RGB, normalizes pixel values,
    and expands dimensions to create a batch of 1.
    """
    img_resized = pil_img.resize(size).convert('RGB')
    img_array = np.array(img_resized) / 255.0
    return np.expand_dims(img_array, axis=0)

def draw_scaled_boxes_on_image(image_np, detections, original_dims, model_dims, conf_threshold=0.5):
    """
    Draws bounding boxes on a numpy image array.
    It scales the box coordinates from the model's input dimensions to the original image's dimensions.
    """
    original_width, original_height = original_dims
    model_width, model_height = model_dims

    # Calculate the scaling factor for both width and height
    x_scale = original_width / model_width
    y_scale = original_height / model_height

    image_with_boxes = image_np.copy()

    # Ensure detections is always a 2D array. If the model returns a single
    # detection, it might be a 1D array (shape (5,)). We reshape it to (1, 5)
    # so the loop below works correctly and consistently.
    if isinstance(detections, np.ndarray) and detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    for det in detections:
        if len(det) == 5:
            # Coordinates from the model are relative to the 224x224 input
            x1_model, y1_model, x2_model, y2_model, conf = det
            
            if conf >= conf_threshold:
                # Scale the coordinates to the original image size
                x1 = int(x1_model * x_scale)
                y1 = int(y1_model * y_scale)
                x2 = int(x2_model * x_scale)
                y2 = int(y2_model * y_scale)

                # Draw the rectangle on the image
                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 3) # Green box
                
                # Add a label with the confidence score
                label = f'{conf:.2f}'
                cv2.putText(image_with_boxes, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2) # Blue text
    
    return image_with_boxes

def encode_image_to_base64(image_np):
    """
    Encodes a numpy image array (in RGB format) to a base64 string for web display.
    """
    # PIL opens images in RGB, but OpenCV's imencode expects BGR.
    # We must convert the color space before encoding.
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', image_bgr)
    return base64.b64encode(buffer).decode('utf-8')

# --- Routes ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Open the image from the file stream once. This is more efficient.
        pil_img = Image.open(file.stream).convert('RGB')

        # --- Classification ---
        class_input = preprocess_image(pil_img, MODEL_INPUT_SIZE)
        malignancy_prob = float(model_classification.predict(class_input)[0][0])
        response = {'malignant_probability': malignancy_prob}

        # --- Detection (if necessary) ---
        if malignancy_prob > DETECTION_THRESHOLD:
            # Preprocess the same image for the detection model
            detection_input = preprocess_image(pil_img, MODEL_INPUT_SIZE)
            
            # Run detection. The output coordinates will be for a 224x224 image.
            detections = model_detection.predict(detection_input)[0]
            
            # --- DEBUGGING STEP ---
            # Log the raw detection output to the console to see what the model is returning.
            # This helps diagnose issues with the model's output format or confidence scores.
            print(f"DEBUG: Raw Detections from Model: {detections}")

            response['detection_triggered'] = True
            response['detection_results'] = detections.tolist()

            # --- Draw boxes on the ORIGINAL full-sized image ---
            original_dims = pil_img.size
            img_np_original = np.array(pil_img)

            # Use our drawing function, but with a very low confidence threshold (0.1)
            # to ensure that if the model finds *anything*, we try to draw it.
            # You can increase this value later (e.g., to 0.5) in production.
            img_with_boxes = draw_scaled_boxes_on_image(img_np_original, detections, original_dims, MODEL_INPUT_SIZE, conf_threshold=0.1)
            
            # Encode the result for the frontend
            response['annotated_image_base64'] = encode_image_to_base64(img_with_boxes)
        else:
            response['detection_triggered'] = False

        return jsonify(response)

    except Exception as e:
        # Log the full error to the server console for easier debugging
        app.logger.error(f"An error occurred during prediction: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# --- Run ---
if __name__ == '__main__':
    app.run(debug=True)
