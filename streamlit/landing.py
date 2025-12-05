import streamlit as st

st.set_page_config(page_title="ML2 Final Project", layout="wide")

st.title("Bayesian Gene Modeling and CNN-Based Cancer Classification")


st.header("Project overview")
st.markdown("""
We studied cancer using two kinds of data and two models.

- A Bayesian regression model uses gene expression to estimate tumor purity
  in lung adenocarcinoma samples.
- A convolutional neural network (CNN) uses histology images to classify tumors
  as cancerous or benign, and into more detailed cancer types.
""")


st.header("Data")
st.markdown("""
Gene expression and tumor purity

- We used the TCGA LUAD cohort from the UCSC Xena portal.  
- Clinical data provided a continuous tumor purity score for each sample
  (percentage of cancer cells versus normal or immune cells).  
- RNA-seq data gave normalized expression for thousands of genes.  
- We selected 25 biologically relevant genes linked to immune activity,
  proliferation, and lung cancer driver pathways, and kept 202 samples
  with both purity and expression.

Cancer images

- For the CNN we used a public lung and colon histopathology dataset.  
- Images cover five classes: lung benign, lung adenocarcinoma, lung squamous
  cell carcinoma, colon benign, and colon adenocarcinoma.  
- We resized images to 128×128, converted them to grayscale, and split
  them into training, validation, and test sets.
""")


st.header("Methods")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Bayesian gene model")
    st.markdown("""
    - Outcome: continuous tumor purity.  
    - Inputs: expression of the 25 selected genes.  
    - Likelihood: Gaussian linear regression with Normal noise.  
    - Priors: weak Normal priors on each coefficient 
      so the data can drive the estimates.  
    - Because Normal prior and Normal likelihood are conjugate, we used the
      closed-form posterior to get posterior means, standard deviations,
      and 95 percent credible intervals for each gene effect.
    """)

with col2:
    st.subheader("CNN image classifier")
    st.markdown("""
    - Architecture: three convolutional layers (16, 32, 64 filters) with 3×3 kernels
      and max-pooling after each layer.  
    - Followed by two dense layers (128 and 64 units) and a final output layer:
      softmax for the 5-way classifier and sigmoid for the binary classifier.  
    - Activation: ReLU for all hidden layers.  
    - Regularization: data augmentation (random flips, rotations, zoom) plus
      two dropout layers to reduce overfitting.
    """)


st.header("Results")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Bayesian model")
    st.markdown("""
    - Train RMSE about 0.11, test RMSE about 0.14.  
    - Train R2 about 0.58, test R2 about 0.54, so the model explains a bit more
      than half of the variance in tumor purity.  
    - Predictive intervals had roughly 95 percent coverage
    - Some genes, like TOP2A, showed a strong positive effect on purity
      (more proliferating tumor cells).  
    - Others, like COL3A1, showed a strong negative effect (more stromal tissue).  
    - Many credible intervals crossed zero, showing high uncertainty in most genes.
    """)

with col4:
    st.subheader("CNN classifier")
    st.markdown("""
    Five-class model

    - Accuracy about 94 percent, low loss around 0.14, F1 score about 94 percent.  
    - The model separates five tumor types much better than random guessing.

    Binary cancer vs benign model

    - Accuracy about 99 percent, loss around 0.03, F1 score about 99 percent.  
    - Training and validation curves stay close for 100 epochs, suggesting little overfitting.

    These results show that the CNN can reliably distinguish cancerous
    from non-cancerous tissue and also separate specific tumor subtypes.
    """)


st.header("Try the image classifier")
st.markdown("""
On the Image Classifier tab, you can upload a cancer cell image from the dataset.

The app will:
1. Convert the image to grayscale and resize it to the model’s input size.  
2. Run it through the trained CNN.  
3. Show the model’s prediction: whether the image looks benign or tumorous.

Switch to the Image Classifier page in the sidebar to see how the model would
label your image.
""")
