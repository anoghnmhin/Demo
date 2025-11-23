import streamlit as st
import numpy as np
import gdown
import os
import pandas as pd
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow import keras

# ----------------------------
# 0. Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Emotion AI",
    page_icon="🎭",
    layout="centered"
)

# ----------------------------
# 1. Link Google Drive
# ----------------------------
MODEL_URL = "https://drive.google.com/uc?id=13RJB6HPpb_0Mx7qoPY8l-g5MzQvvU9Nd"
MODEL_PATH = "final_vgg16_affectnet.keras"

# ----------------------------
# 2. Download Model
# ----------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (~160MB), please wait..."):
            try:
                gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
                st.success("Model downloaded successfully!")
            except Exception as e:
                st.error(f"Error downloading model: {e}")

download_model()

# ----------------------------
# 3. Load Model with Cache
# ----------------------------
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ----------------------------
# 4. Emotion Labels & Assets
# ----------------------------
emotion_classes = [
    'anger', 'contempt', 'disgust', 'fear',
    'happy', 'neutral', 'sad', 'surprise'
]

# Map emotions to emojis for better UI
emoji_map = {
    'anger': '😡', 'contempt': '😒', 'disgust': '🤢', 'fear': '😱',
    'happy': '😊', 'neutral': '😐', 'sad': '😢', 'surprise': '😲'
}

# ----------------------------
# 5. Predict Logic (Unchanged)
# ----------------------------
def predict_emotion(img):
    # Preprocessing logic kept exactly as requested
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    img_preprocessed = preprocess_input(img_array)
    img_expanded = np.expand_dims(img_preprocessed, axis=0)

    preds = model.predict(img_expanded)[0]
    label_index = np.argmax(preds)
    confidence = preds[label_index]

    return emotion_classes[label_index], confidence, preds

# ----------------------------
# 6. UI Implementation
# ----------------------------

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/10603/10603327.png", width=100)
    st.title("About App")
    st.info(
        """
        This app uses a **VGG16** model fine-tuned on the **AffectNet** dataset.
        
        **Classes:**
        - Anger 😡
        - Contempt 😒
        - Disgust 🤢
        - Fear 😱
        - Happy 😊
        - Neutral 😐
        - Sad 😢
        - Surprise 😲
        """
    )
    st.write("---")
    st.caption("Powered by TensorFlow & Streamlit")

# Main Content
st.title("🎭 Facial Emotion Recognition")
st.markdown("### Upload a portrait to analyze emotions")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Layout: Image on the left, Results on the right (after button press)
    col1, col2 = st.columns([1, 1])
    
    image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Button to trigger prediction
    analyze_button = st.button("🔍 Analyze Emotion", use_container_width=True)

    if analyze_button:
        if model is None:
            st.error("Model failed to load. Please check the download.")
        else:
            with st.spinner("Analyzing..."):
                # Get prediction
                label, conf, all_preds = predict_emotion(image)
                
                # Display Results in the second column
                with col2:
                    st.markdown("### Result")
                    
                    # Primary Metric
                    st.metric(
                        label=f"Dominant Emotion",
                        value=f"{emoji_map.get(label, '')} {label.title()}",
                        delta=f"{conf*100:.1f}% Confidence"
                    )
                    
                    # Progress bar for the top result
                    st.progress(float(conf))

            # Show Detailed Probability Chart below
            st.divider()
            st.subheader("📊 Probability Distribution")
            
            # Create a DataFrame for the chart
            df_probs = pd.DataFrame({
                'Emotion': [e.title() for e in emotion_classes],
                'Probability': all_preds
            })
            
            # Display Bar Chart
            st.bar_chart(df_probs.set_index('Emotion'), color="#FF4B4B")
