import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np

#load the trained model
model = tf.keras.models.load_model("model.h5")

#load the encoders and scaler
with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open("onehot_encoder_geo.pkl", "rb") as f:
    onehot_encoder_geo = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

#streamlit app
st.title("CUSTOMER PREDICTION CHURN")

#user input
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

#prepare the input data
input_data = {
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [label_encoder_gender.transform([gender])[0]], #label encoder
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
}

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]])

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Convert dictionary to DataFrame
input_df = pd.DataFrame(input_data)

# Drop original Geography column
input_df = input_df.drop('Geography', axis=1)

# Add one-hot encoded columns
input_df = pd.concat([input_df, geo_encoded_df], axis=1)

# Reorder columns exactly like during training
input_df = input_df.reindex(columns=scaler.feature_names_in_)

# Scale
input_scaled = scaler.transform(input_df)

#predict the churn
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

if prediction_proba > 0.5:
    st.write("The customer is likely to churn")
else:
    st.write("The customer is not likely to churn")

