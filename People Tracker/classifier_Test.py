import tensorflow as tf
import numpy as np
import cv2

# Load the trained model

gender_classifier = tf.keras.models.load_model("C:\\Users\\USER\\Documents\\Work\\Models\\gender_classifier_model.h5")

# Define image preprocessing function
def preprocess_image(image_path, image_size=(128, 128)):
    img = cv2.imread(image_path)  # Load image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, image_size)  # Resize to match model input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Test with a sample image
image_path = "C:\\Users\\USER\\.cache\\kagglehub\\datasets\\cashutosh\\gender-classification-dataset\\versions\\1\\Validation\\male\\linkedin pp.jpg"  # Change this to your test image path
face_img = preprocess_image(image_path)

# Make prediction
prediction = gender_classifier.predict(face_img)[0][0]  # Get the first output value

# Interpret result
gender = "Female" if prediction < 0.5 else "Male"
confidence = (1 - prediction) if prediction < 0.5 else prediction

print(f"Predicted Gender: {gender} ({confidence:.2%} confidence)")

