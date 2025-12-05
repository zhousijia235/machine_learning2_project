import streamlit as st
import cv2
import pandas as pd
import numpy as np
from keras.models import load_model
import os

# Load the models and create class definitions
IMAGE_DIR = 'sample_images'
binary_model = load_model('models/binary_cnn.keras')
cat_model = load_model('models/categorical_cnn.keras')
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

# The beginning of the page starts here
st.title('Interactive Cancer Classifier')

uploaded_image = st.file_uploader('Choose a file', type=['png', 'jpeg', 'jpg'])

# check for when the file is uploaded
if uploaded_image is not None:
    try:
        image = preprocess_image(uploaded_image)
        predict(image, True)
    except Exception as e:
        st.error(f"Error reading file: {e}")

# providing test images as well, don't expect people to have random histopathology images
images_from_folder = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
selected_image_file = st.selectbox('Choose a sample tumor image:', images_from_folder)
# Load the selected image
if selected_image_file:
    image_path = os.path.join(IMAGE_DIR, selected_image_file)
    select_image = cv2.imread(image_path)
    st.image(select_image, caption=selected_image_file, width=256)

# Predict if sample image used
if st.button("Run Prediction"):
    try:
        select_image = preprocess_sample(select_image)
        predict(select_image, False)
    except Exception as e:
        st.error(f"Error reading file: {e}")