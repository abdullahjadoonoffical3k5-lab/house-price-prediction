import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

st.title("🏠 House Price Prediction App")
st.write("Apne ghar ki details enter karein aur estimated price dekhein.")

try:
    model = joblib.load("house_price_model.pkl")

    area = st.number_input("Area (square feet)", min_value=100, value=1800, step=50)
    bedrooms = st.number_input("Bedrooms", min_value=1, value=4, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=1, value=3, step=1)
    age_years = st.number_input("House age (years)", min_value=0, value=2, step=1)

    if st.button("Predict Price"):
        new_house = pd.DataFrame({
            "area_sqft": [area],
            "bedrooms": [bedrooms],
            "bathrooms": [bathrooms],
            "age_years": [age_years]
        })


        predicted_price = model.predict(new_house)[0]
        st.success(f"Estimated House Price: {predicted_price:.2f} lakh")
        st.caption("Yeh learning demo hai; estimate real market price ki guarantee nahi hai.")

except FileNotFoundError:
    st.error("house_price_model.pkl file nahi mili. Pehle notebook se trained model save karein.")
