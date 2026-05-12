import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

def get_file_path(filename):
    """Get the absolute or relative path to a file, handling both local and Streamlit Cloud."""
    # Try current directory first
    if os.path.exists(filename):
        return filename
    # Try relative to script directory
    script_dir = Path(__file__).parent
    script_path = script_dir / filename
    if script_path.exists():
        return str(script_path)
    # Fallback to filename (works on Streamlit Cloud)
    return filename

@st.cache_resource
def load_model_and_preprocessors():
    """Load model and preprocessors with error handling."""
    try:
        # Get file paths
        model_path = get_file_path('churn_model.h5')
        
        # Check if model file exists
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            st.info("Please ensure churn_model.h5 is in the repository")
            st.stop()
        
        # Load model with custom object scope
        model = tf.keras.models.load_model(model_path)
        
        # Load scaler
        scaler_path = get_file_path('scaler.pkl')
        if not os.path.exists(scaler_path):
            st.error(f"❌ Scaler file not found: {scaler_path}")
            st.info("Please ensure scaler.pkl is in the repository")
            st.stop()
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load one-hot encoder
        encoder_path = get_file_path('one_hot_encoder.pkl')
        if not os.path.exists(encoder_path):
            st.error(f"❌ One-hot encoder file not found: {encoder_path}")
            st.info("Please ensure one_hot_encoder.pkl is in the repository")
            st.stop()
        
        with open(encoder_path, 'rb') as f:
            one_hot_encoder = pickle.load(f)
        
        # Load label encoder
        label_encoder_path = get_file_path('label_encoder.pkl')
        if not os.path.exists(label_encoder_path):
            st.error(f"❌ Label encoder file not found: {label_encoder_path}")
            st.info("Please ensure label_encoder.pkl is in the repository")
            st.stop()
        
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        return model, scaler, one_hot_encoder, label_encoder
    
    except Exception as e:
        st.error(f"❌ Error loading files: {str(e)}")
        st.stop()

# Load all resources
model, scaler, one_hot_encoder, label_encoder = load_model_and_preprocessors()

# Streamlit app
st.title("🔮 Customer Churn Prediction")
st.write("Enter customer details to predict churn probability.")
st.markdown("---")

# Create two columns for better UI
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('Geography', one_hot_encoder.categories_[0])
    gender = st.selectbox('Gender', label_encoder.classes_)
    age = st.slider('Age', 18, 90, 35)
    balance = st.number_input('Balance ($)', min_value=0.0, value=50000.0)

with col2:
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
    estimated_salary = st.number_input('Estimated Salary ($)', min_value=0.0, value=50000.0)
    tenure = st.slider('Tenure (years)', 0, 10, 5)
    num_of_products = st.slider('Number of Products', 1, 4, 1)

col3, col4 = st.columns(2)

with col3:
    has_cr_card = st.selectbox('Has Credit Card', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

with col4:
    is_active_member = st.selectbox('Is Active Member', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

st.markdown("---")

# Prediction button
if st.button('🔍 Predict Churn', use_container_width=True):
    try:
        # Prepare input data with correct column names matching scaler.feature_names_in_
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
        geography_df = pd.DataFrame(
            geography_encoded,
            columns=one_hot_encoder.get_feature_names_out(['Geography'])
        )
        
        # Combine all features
        input_data = pd.concat([input_data, geography_df], axis=1)
        
        # Reorder columns to match scaler.feature_names_in_
        expected_features = scaler.feature_names_in_
        input_data = input_data[expected_features]
        
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction_prob = model.predict(input_scaled, verbose=0)[0][0]
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        col_result1, col_result2 = st.columns(2)
        
        with col_result1:
            st.metric(
                "Churn Probability",
                f"{prediction_prob:.2%}",
                delta=None
            )
        
        with col_result2:
            churn_status = "⚠️ High Risk" if prediction_prob > 0.5 else "✅ Low Risk"
            st.metric("Status", churn_status)
        
        # Detailed message
        st.markdown("---")
        if prediction_prob > 0.5:
            st.warning(
                f"⚠️ **High Churn Risk**: The customer has a {prediction_prob:.2%} probability of churning. "
                "Consider retention strategies."
            )
        else:
            st.success(
                f"✅ **Low Churn Risk**: The customer has only a {prediction_prob:.2%} probability of churning. "
                "Customer appears stable."
            )
    
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        st.info("Please ensure all input values are valid and try again.")