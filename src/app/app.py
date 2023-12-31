# Import necessary libraries and modules
import os                # Provides access to operating system functions
from fastapi import FastAPI, File, UploadFile  # Imports classes and functions from FastAPI framework
from dotenv import load_dotenv  # Loads environment variables from a .env file
from model import preprocess_image, predict

# Load environment variables from a .env file in the project directory
load_dotenv()
# Create an instance of the FastAPI framework
app = FastAPI()
# Retrieve the external API URL from the environment variables
API_URL = os.getenv("API_URL")

# Define a FastAPI route to handle POST requests at "/predict/"
@app.post("/predict/")
async def predict_handler(file: UploadFile = File(...)):
    # Read the content of the uploaded file asynchronously
    content = await file.read()
    # Preprocess the image
    image = preprocess_image(content)
    # Make predictions using the model
    predictions = predict(image)
    return {"predictions": predictions}



