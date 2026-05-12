import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_model_and_preprocessors():
    """Load model and preprocessors with error handling."""
    try:
        # Load model
        model = tf.keras.models.load_model('churn_model.h5')
        
        # Load scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load encoders
        with open('one_hot_encoder.pkl', 'rb') as f:
            one_hot_encoder = pickle.load(f)
        
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        return model, scaler, one_hot_encoder, label_encoder
    
    except FileNotFoundError as e:
        st.error(f"❌ Missing file: {str(e)}")
        st.info("Required files: churn_model.h5, scaler.pkl, one_hot_encoder.pkl, label_encoder.pkl")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Load resources
model, scaler, one_hot_encoder, label_encoder = load_model_and_preprocessors()

# App title and description
st.title("🔮 Customer Churn Prediction")
st.markdown("Enter customer details to predict churn probability")
st.divider()

# Create input columns
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('Geography', one_hot_encoder.categories_[0])
    gender = st.selectbox('Gender', label_encoder.classes_)
    age = st.slider('Age', 18, 90, 35)
    balance = st.number_input('Balance ($)', min_value=0.0, value=50000.0)

with col2:
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
    estimated_salary = st.number_input('Estimated Salary ($)', min_value=0.0, value=75000.0)
    tenure = st.slider('Tenure (years)', 0, 10, 5)
    num_of_products = st.slider('Number of Products', 1, 4, 1)

col3, col4 = st.columns(2)
with col3:
    has_cr_card = st.selectbox('Has Credit Card', [0, 1], format_func=lambda x: 'Yes' if x else 'No')
with col4:
    is_active_member = st.selectbox('Is Active Member', [0, 1], format_func=lambda x: 'Yes' if x else 'No')

st.divider()

# Prediction button
if st.button('🔍 Predict Churn', use_container_width=True, type="primary"):
    try:
        # Prepare input data
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })
        
        # One-hot encode geography
        geography_encoded = one_hot_encoder.transform([[geography]]).toarray()
        geography_cols = one_hot_encoder.get_feature_names_out(['Geography'])
        geography_df = pd.DataFrame(geography_encoded, columns=geography_cols)
        
        # Combine features
        input_data = pd.concat([input_data, geography_df], axis=1)
        
        # Reorder columns to match scaler
        expected_features = scaler.feature_names_in_
        input_data = input_data[expected_features]
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction_prob = model.predict(input_scaled, verbose=0)[0][0]
        
        # Display results
        st.divider()
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.metric("Churn Probability", f"{prediction_prob:.2%}")
        
        with col_res2:
            status = "⚠️ High Risk" if prediction_prob > 0.5 else "✅ Low Risk"
            st.metric("Status", status)
        
        st.divider()
        
        # Detailed message
        if prediction_prob > 0.5:
            st.warning(f"⚠️ **High Churn Risk** ({prediction_prob:.2%}) - Consider retention strategies")
        else:
            st.success(f"✅ **Low Churn Risk** ({prediction_prob:.2%}) - Customer appears stable")
    
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        st.info("Please check your inputs and try again")