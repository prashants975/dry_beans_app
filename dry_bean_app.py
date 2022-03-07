import pandas as pd
import streamlit as st
import pickle


st.header("Dry Beans Classification!!")
st.write("""
This is a demo application for the Dry Beans Classification Project.
Enter the values of the features to get the classification for the beans.
""")

st.image("Beans.jpg")

# load the model from disk
scaler = pickle.load(open("scaler.sav", 'rb'))
model = pickle.load(open("finalized_model.sav", 'rb'))
print("Model Loaded")



with st.form("my_form"):
    st.write("Insert the Data for Classification of the Beans:")
    
    Area = st.text_input("Area", value="0.5")
    Perimeter  = st.text_input("Perimeter ", value="0.5")
    MajorAxisLength = st.text_input("Major Axis Length", value="0.5")
    MinorAxisLength = st.text_input("Minor Axis Length", value="0.5")
    AspectRatio = st.text_input("AspectRatio", value="0.5")
    Eccentricity = st.text_input("Eccentricity", value="0.5")
    ConvexArea = st.text_input("ConvexArea", value="0.5")
    EquiDiameter = st.text_input("EquiDiameter", value="0.5")
    Extent = st.text_input("Extent", value="0.5")
    Solidity = st.text_input("Solidity", value="0.5")
    Roundness = st.text_input("Roundness", value="0.5")
    Compactness = st.text_input("Compactness", value="0.5")
    ShapeFactor1 = st.text_input("Shape Factor 1", value="0.5")
    ShapeFactor2 = st.text_input("Shape Factor 2", value="0.5")
    ShapeFactor3 = st.text_input("Shape Factor 3", value="0.5")
    ShapeFactor4 = st.text_input("Shape Factor 4", value="0.5")

    features = [
            Area,
            Perimeter,
            MajorAxisLength,
            MinorAxisLength,
            AspectRatio,
            Eccentricity,
            ConvexArea,
            EquiDiameter,
            Extent,
            Solidity,
            Roundness,
            Compactness,
            ShapeFactor1,
            ShapeFactor2,
            ShapeFactor3,
            ShapeFactor4,
        ]




    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        X = scaler.transform([features])
        y = model.predict(X)
        st.write("The type of Beans is:", y[0])


