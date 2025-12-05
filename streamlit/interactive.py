import streamlit as st
import cv2
import pandas as pd
import numpy as np
from keras.models import load_model
import os
from pathlib import Path

# Load the models and create class definitions

BASE_DIR = Path(__file__).parent
IMAGE_DIR = BASE_DIR / "sample_images"
BINARY_MODEL_PATH = BASE_DIR / "models" / "binary_cnn.keras"
CAT_MODEL_PATH = BASE_DIR / "models" / "categorical_cnn.keras"

binary_model = load_model(str(BINARY_MODEL_PATH))
cat_model = load_model(str(CAT_MODEL_PATH))
cat_classes = ['colon adenocarcinoma (cancerous)', 
                       'colon benign', 
                       'lung adenocarcinoma (cancerous)', 
                       'lung benign', 
                       'lung squamous (cancerous)']

# preprocess uploaded image
def preprocess_image(uploaded_img):
    img_array = np.asarray(bytearray(uploaded_img.read()), dtype='uint8')
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

# preprocess sample image
def preprocess_sample(img):
    image = cv2.resize(img, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def predict(image, isUpload: bool):
    if isUpload:
        st.write("File uploaded successfully!")
    st.image(image, caption='your selecgted tumor image', width=256)
    # change the shape from (128, 128) -> (1, 128, 128, 1)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    pred = binary_model.predict(image)
    pred = (pred > 0.5).astype(int).flatten()
    print(pred)
    # Predict if cancerous or not
    if pred[0] == 0:
        st.write("The binary model predicts that this image is a cancerous tumor")
    else:
        st.write("The binary model predicts that this image is a benign tumor")
    
    # Predict the specific type of cancer
    pred = cat_model.predict(image).flatten()
    pred = np.argmax(pred)
    print(pred)
    st.write("The categorical model predicts that this image is ", cat_classes[pred])

# page content
st.title('Interactive Cancer Classifier')

st.markdown("""
Welcome to our cancer histopathology image classifier demo! 

st.image(
    "https://img.freepik.com/free-photo/cute-little-dog-impersonating-business-person_23-2148985938.jpg?semt=ais_hybrid&w=740&q=80",
    caption="Your friendly cancer-detecting assistant",
    use_column_width=True
)


This page lets you try a convolutional neural network (CNN) that reaches about 99% accuracy
at telling cancer histology images from benign ones. 
But don't take our word, try it your self! 

**How to use it**

- Upload a lung or colon histopathology tile, or pick one of the sample images below.  
- The binary model will first decide whether the image looks **cancerous or benign**.  
- If the tissue is lung or colon, the categorical model will also predict the specific type  
  (for example lung adenocarcinoma, lung squamous, or colon adenocarcinoma).
""")

icon_image = IMAGE_DIR / "lungn38.jpeg"
if icon_image.exists():
    st.image(str(icon_image), width=120, caption="example histopathology tile")



#upload
uploaded_image = st.file_uploader('Choose a file', type=['png', 'jpeg', 'jpg'])

# check for when the file is uploaded
if uploaded_image is not None:
    try:
        image = preprocess_image(uploaded_image)
        predict(image, True)
    except Exception as e:
        st.error(f"Error reading file: {e}")

# providing test images as well, don't expect people to have random histopathology images
if IMAGE_DIR.exists():
    images_from_folder = [
        f.name for f in IMAGE_DIR.iterdir()
        if f.suffix.lower() in ('.png', '.jpg', '.jpeg')
    ]
else:
    st.warning("Sample_images folder not found on the server.")
    images_from_folder = []

selected_image_file = st.selectbox('Choose a sample tumor image:', images_from_folder)

if selected_image_file:
    image_path = IMAGE_DIR / selected_image_file
    select_image = cv2.imread(str(image_path))
    st.image(select_image, caption=selected_image_file, width=256)
else:
    select_image = None

# Predict if sample image used
if st.button("Run Prediction"):
    try:
        select_image = preprocess_sample(select_image)
        predict(select_image, False)
    except Exception as e:
        st.error(f"Error reading file: {e}")
