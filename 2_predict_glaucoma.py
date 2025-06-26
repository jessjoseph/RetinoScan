# 2_predict_glaucoma.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore

# Load the trained model
model = load_model('glaucoma_model.h5')

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))         # Resize to match training input
    image = img_to_array(image) / 255.0           # Convert to array and normalize
    image = np.expand_dims(image, axis=0)         # Add batch dimension
    return image

# Path to the image you want to test
image_path = input("Enter the path to the eye image: ")
image = preprocess_image(image_path)

# Make prediction
prediction = model.predict(image)
predicted_class = int(prediction[0][0] > 0.5)

# Show result
if predicted_class == 1:
    print("Prediction: Glaucoma Positive")
else:
    print("Prediction: Glaucoma Negative")
