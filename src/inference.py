# inference.py
import os
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image


def load_efficientnet_model(model_path, weights_path):
    model = load_model(model_path)
    model.load_weights(weights_path)
    return model


def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Adjust the size based on your model's input size
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def predict_age(model, image_array):
    prediction = model.predict(image_array)
    return prediction



