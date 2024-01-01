# app.py
import streamlit as st
from inference import load_efficientnet_model, preprocess_image, predict_age
import numpy as np
from google.colab import drive


# Load the EfficientNet model and corresponding weights
drive.mount('/content/drive')
# model_path = "/content/drive/MyDrive/EfficientNet-91.80.h5"
# weights_path = "/content/drive/MyDrive/EfficientNet-weights.h5"
model_path = "../models/EfficientNet-91.80.h5"
weights_path = "../models/EfficientNet-weights.h5"
model = load_efficientnet_model(model_path, weights_path)

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
        image_array = preprocess_image(input_image)

        # Get the prediction from the model
        prediction = predict_age(model, image_array)

        # Display the uploaded image
        st.image(input_image, caption='Uploaded Image.', use_column_width=True)
        st.write('Please wait while predicting...')

        # Display prediction results
        labels = ["Class 0", "Class 1"]  # Update with your actual class labels
        st.bar_chart(prediction[0])
        st.write("Predicted Label:", labels[np.argmax(prediction[0])])
    else:
        st.write('Insert an image for prediction!')


