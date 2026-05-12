import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
# import the model
model =tf.keras.models.load_model('churn_model.h5')

# load the encoders and scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('one_hot_encoder.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
    
# streatmlit app
st.title("Customer Churn Prediction")
st.write("Enter the customer details to predict the churn probability.")

# input fields

geograhpy = st.selectbox('Geography',one_hot_encoder.categories_[0])
gender = st.selectbox('Gender',label_encoder.classes_)
age=st.slider('Age',18,90)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card', [0,1])
is_active_member=st.selectbox('Is Active Member', [0,1])

# prepare input data
input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
    
})
# one hot encoder geography
geography_encoded=one_hot_encoder.transform([[geograhpy]]).toarray()
geography_df=pd.DataFrame(geography_encoded,columns=one_hot_encoder.get_feature_names_out())
input_data=pd.concat([input_data.reset_index(drop=True),geography_df],axis=1)
# scale the input data
input_scaled=scaler.transform(input_data)

# predict churn
prediction_churn =model.predict(input_scaled)[0][0]
st.write(f"Churn Probability: {prediction_churn:.4f}")
if prediction_churn > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is unlikely to churn.")