import streamlit as st


st.title("Bayesian Gene Modeling and CNN Based Cancer Classification")



st.set_page_config(page_title="ML2 Final Project", layout="wide")

st.title("Bayesian Gene Modeling and CNN-Based Cancer Classification")

st.markdown("""
### What this project is about

We wanted to study cancer from **two angles**:


1. **Cancer image classification**  
   We train a **convolutional neural network (CNN)** on cancer images to predict
   image-level labels (for example tumor vs normal or cancer subtype).
   This shows how image features can support or complement the gene-level results.

2. **Gene expression and tumor purity**  
   We use a Bayesian model to estimate how the expression of individual genes
   is related to **tumor purity** (how much of a sample is tumor cells vs normal cells).
   The Bayesian approach gives us both an effect size and **uncertainty** for each gene.


Together, these two methods connect **molecular data** (gene expression)
with **visual patterns** in cancer images.
""")

st.markdown("---")

st.subheader("Overview of models")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Bayesian modeling**  
    - Cleaned and filtered the gene expression and tumor purity data  
    - Fit a Bayesian regression model for gene effect on tumor purity  
    - Summarized posterior means and credible intervals for selected genes  
    """)

with col2:
    st.markdown("""
    **CNN classification **  
    - Prepared and augmented cancer image data  
    - Trained a CNN to classify images into cancer images into diagnostic categories
    - Evaluated performance and visualized example predictions  
    """)

st.markdown("---")

st.subheader("How to use this app")

st.markdown("""
 
- **Image Classifier:** upload a cancer cell image and see whether the CNN predicts it
  as **benign or tumorous**.
""")
















