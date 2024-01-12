# Project Sport Classifier

This project includes code for model inference and a simple UI created using Streamlit to visualize the inference of the trained best model.

## Repository Contents

- `inference.py`: Code for model inference.
- `app.py`: Code for creating a simple UI using Streamlit to visualize the inference of the trained best model.

## Instructions for Running

1. **Add model weight**
   Download file "weights_EfficientNet_best.h5" from
   https://drive.google.com/file/d/1U67VR7y0S6xFurOEAxLLJ9gu8ZmSZl04/view?usp=sharing
   and save it at folder "models"

2. **Install Dependencies**

   Make sure you have the required dependencies installed. Use a virtual environment if needed:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run Inference**

   Run the code for model inference:

   ```bash
   python src/inference.py
   ```

4. **Run UI**

   Run the code for creating a UI using Streamli:

   ```bash
   streamlit run src/app.py
   ```
