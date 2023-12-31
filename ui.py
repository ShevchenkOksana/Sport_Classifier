# Import necessary libraries and modules
import os
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
import numpy as np

# Load environment variables from a .env file in the project directory
load_dotenv()

# Constants - Retrieve configuration from environment variables
API_URL = os.getenv("API_URL")  # External API URL
PREDICTION_ENDPOINT = os.getenv("PREDICTION_ENDPOINT")  # Endpoint for predictions

# Load the EfficientNet model and corresponding weights
model_path = "EfficientNet-91.80.h5"
weights_path = "EfficientNet-weights.h5"
model = load_model(model_path)
model.load_weights(weights_path)

# Streamlit configuration
st.set_page_config(layout="wide")
st.markdown('<h1 style="text-align: center; color: #fffff;">EfficientNet Age Classifier</h1>',
            unsafe_allow_html=True)

# Display introductory text
st.write(
    '''
    This is a simple app for the classification of EfficientNet Age Classifier.
    This Streamlit example uses a FastAPI service as the backend.
    Visit this URL at `http://0.0.0.0:8000/docs` for FastAPI documentation.
    '''
)

# File uploader widget
input_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

# Prediction button
if st.button('Get Prediction'):
    if input_image is not None:
        # Preprocess the input image for model prediction
        image = Image.open(input_image)
        image = image.resize((224, 224))  # Adjust the size based on your model's input size
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Get the prediction from the model
        prediction = model.predict(image_array)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write('Please wait while predicting...')

        # Display prediction results
        labels = ["Class 0", "Class 1"]  # Update with your actual class labels
        st.bar_chart(prediction[0])
        st.write("Predicted Label:", labels[np.argmax(prediction[0])])
    else:
        st.write('Insert an image for prediction!')
