import streamlit as st
from keras.models import load_model

st.title("Interactive Classifier")

# Load the model
model = load_model('../models/binary_cnn.keras')
