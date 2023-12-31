from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO

# Load the model
model = load_model("EfficientNet-91.80.h5")


def preprocess_image(file_content):
    # Load and preprocess the image
    image = Image.open(BytesIO(file_content))
    # Convert image to numpy array
    img_array = np.array(image)
    return img_array


def predict(image):
    # Perform inference using the loaded model
    prediction = model.predict(np.expand_dims(image, axis=0))
    return prediction.tolist()
