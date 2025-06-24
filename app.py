import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
import requests

# Page setup
st.set_page_config(page_title="Deepfake Detector", page_icon="🔍", layout="wide")

# Title
st.title("🔍 AI Deepfake Detection")
st.write("Upload a face image to check if it's real or fake!")

# Download model from Google Drive
def download_model():
    model_path = 'deepfake_model_densenet121.h5'
    
    if not os.path.exists(model_path):
        st.info("📥 Downloading AI model... (this takes 1-2 minutes first time)")
        
        # REPLACE THIS URL WITH YOUR GOOGLE DRIVE LINK
        # Change the end from /view?usp=sharing to /uc?export=download
        drive_url = "https://drive.google.com/file/d/1hrY-nAoiL7jE1i7Qo2twLS2fcRbOIA7g/view?usp=sharing"
        
        try:
            response = requests.get(drive_url)
            with open(model_path, 'wb') as f:
                f.write(response.content)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Error downloading model: {e}")
            st.error("Please check your Google Drive link!")
            return None
    
    return model_path

# Load model
@st.cache_resource
def load_model():
    try:
        model_path = download_model()
        if model_path is None:
            return None
            
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def predict_image(model, image):
    try:
        # Resize image to 256x256
        image = image.resize((256, 256))
        
        # Convert to array and normalize
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array)[0][0]
        
        # Calculate percentages
        fake_percent = prediction * 100
        real_percent = (1 - prediction) * 100
        
        return fake_percent, real_percent
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

# Main app
def main():
    # Load model
    model = load_model()
    
    if model is None:
        st.error("❌ Cannot load model. Please check the setup.")
        st.stop()
    
    st.success("✅ AI Model loaded successfully!")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Show uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Your Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.subheader("🤖 AI Analysis")
            
            # Analyze button
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing..."):
                    fake_prob, real_prob = predict_image(model, image)
                
                if fake_prob is not None:
                    # Show results
                    if fake_prob > real_prob:
                        st.error("🚨 **FAKE DETECTED!**")
                        st.error(f"Confidence: {fake_prob:.1f}%")
                    else:
                        st.success("✅ **APPEARS REAL**")
                        st.success(f"Confidence: {real_prob:.1f}%")
                    
                    # Show detailed results
                    st.write("**Detailed Results:**")
                    st.write(f"Real: {real_prob:.1f}%")
                    st.write(f"Fake: {fake_prob:.1f}%")
                    
                    # Progress bars
                    st.write("Real Image:")
                    st.progress(real_prob / 100)
                    st.write("Fake Image:")
                    st.progress(fake_prob / 100)

if __name__ == "__main__":
    main()