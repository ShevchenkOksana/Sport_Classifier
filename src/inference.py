# inference.py
import numpy as np
from PIL import Image


def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def predict_age(model, image_array):
    prediction = model.predict(image_array)
    return prediction



