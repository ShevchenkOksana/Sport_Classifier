# app.py
import streamlit as st
from inference import preprocess_image, predict_age
import numpy as np
import torch
from efficientnet_pytorch import EfficientNet


# Instantiate the EfficientNet model
num_labels = 100
efficientnet_variant = 'b3'
Efficient_Net = EfficientNet.from_pretrained(f'efficientnet-{efficientnet_variant}', num_classes=num_labels)
weights_path = "../models/weights_EfficientNet_best.h5"

# Load the weights into the model
state_dict = torch.load(weights_path)
model = Efficient_Net.load_state_dict(state_dict)

# Streamlit configuration
st.set_page_config(layout="wide")
st.markdown('<h1 style="text-align: center; color: #fffff;">EfficientNet Sport Classifier</h1>',
            unsafe_allow_html=True)

# Display introductory text
st.write(
    '''
    This is a simple app for the classification of EfficientNet Sport Classifier.
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


