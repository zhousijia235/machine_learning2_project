import streamlit as st
# add a title for the app

pg = st.navigation([
    st.Page("landing.py", title="Bayesian Gene Modeling and CNN Based Cancer Classification"),
    st.Page("interactive.py", title="Interactive Classifier"),
])
pg.run()