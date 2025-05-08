import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model_path = 'CustomerSegmentationModel_Tuned.pkl'
model = joblib.load(model_path)

# App title
st.title("Customer Segmentation Prediction App")

# Sidebar for user inputs
st.sidebar.header("Customer Attributes")

# Function to get user inputs
def get_user_inputs():
    recency = st.sidebar.number_input("Recency (Days since last purchase)", min_value=0, max_value=365, value=30)
    frequency = st.sidebar.number_input("Frequency (Number of purchases)", min_value=0, value=5)
    monetary = st.sidebar.number_input("Monetary Value (Total Spend)", min_value=0, value=100)
    return pd.DataFrame({
        'Recency': [recency],
        'Frequency': [frequency],
        'Monetary': [monetary]
    })

# Get user inputs
user_data = get_user_inputs()

# Display user inputs
st.subheader("Customer Attributes")
st.write(user_data)

# Predict segment
if st.button("Predict Customer Segment"):
    prediction = model.predict(user_data)
    st.subheader("Predicted Customer Segment")
    st.write(f"Segment: {prediction[0]}")

# Footer
st.markdown("---")
st.write("This app predicts customer segments based on Recency, Frequency, and Monetary values. Tailored marketing strategies can be built based on these insights.")
