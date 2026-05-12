# Customer Churn Prediction App

A Streamlit web app for predicting customer churn using a deep learning model.

## Deployment to Streamlit Cloud

### ✅ Prerequisites
- GitHub account with this repository
- Streamlit Cloud account (free at https://streamlit.io/cloud)

### 🚀 Deployment Steps

1. **Ensure all required files are in Git:**
   ```bash
   git add churn_model.h5
   git add scaler.pkl
   git add one_hot_encoder.pkl
   git add label_encoder.pkl
   git add .streamlit/config.toml
   git add requirements.txt
   git add runtime.txt
   git add app.py
   git commit -m "Add model files and deployment config"
   git push
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to https://streamlit.io/cloud
   - Click "New app"
   - Select your GitHub repository
   - Select the branch: `main` (or your working branch)
   - Set Main file path: `app.py`
   - Click "Deploy"

3. **Wait for deployment to complete** (2-5 minutes)

### 📁 File Structure
```
.
├── app.py                      # Main Streamlit app
├── churn_model.h5              # Trained neural network model
├── scaler.pkl                  # Feature scaler
├── one_hot_encoder.pkl         # Geography encoder
├── label_encoder.pkl           # Gender encoder
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Python version (3.11)
├── .streamlit/
│   └── config.toml             # Streamlit config
└── .gitignore                  # Git ignore rules
```

### ⚙️ Configuration
- **Python Version:** 3.11
- **TensorFlow:** CPU version (2.15.0) for Streamlit Cloud compatibility
- **Memory Optimized:** Yes

### 🐛 Troubleshooting

#### Error: "Oh no. Error running app."
**Solution:** Check Streamlit Cloud logs for specific error:
1. Go to your app on Streamlit Cloud
2. Click "Manage app" → "View logs"
3. Look for error messages
4. Common issues:
   - Model files not in git: `git add *.h5 *.pkl`
   - Wrong Python version: Check `runtime.txt` is `python-3.11`
   - Memory issue: Already optimized with `tensorflow-cpu`

#### Error: "FileNotFoundError: onehot_encoder.pkl"
**Solution:** File doesn't exist or not committed to git
```bash
git add one_hot_encoder.pkl
git commit -m "Add one-hot encoder"
git push
```

#### Error: "Feature names should match"
**Solution:** Already fixed in latest app.py - ensures column order matches scaler

### 📊 Features
- ✅ Customer information input (geography, gender, age, etc.)
- ✅ Deep learning model predictions
- ✅ Churn probability visualization
- ✅ Risk assessment (High/Low)
- ✅ Error handling and validation
- ✅ Deployment-ready code

### 📝 Requirements
- numpy==1.26.4
- pandas==2.2.3
- scikit-learn==1.4.2
- tensorflow-cpu==2.15.0
- tensorboard
- streamlit

### 🔗 Links
- Streamlit Cloud: https://streamlit.io/cloud
- Streamlit Docs: https://docs.streamlit.io
- TensorFlow Docs: https://www.tensorflow.org
