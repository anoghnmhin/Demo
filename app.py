import streamlit as st
import numpy as np
import gdown
import os
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow import keras

# ============================
# Page Config
# ============================
st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="😊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================
# Light & Clean CSS
# ============================
st.markdown("""
<style>
    .main {
        background: #f8fafc;
        padding: 2rem;
    }
    .title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .upload-area {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: #ffffff;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
    }
    .emotion-big {
        font-size: 4.5rem;
        margin: 0.5rem 0;
    }
    .confidence-bar {
        height: 12px;
        background: #e2e8f0;
        border-radius: 6px;
        overflow: hidden;
        margin: 1rem 0;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6, #10b981);
        border-radius: 6px;
        transition: width 1s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# Title & Description
# ============================
st.markdown('<h1 class="title">😊 Facial Emotion Recognition</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a clear portrait photo and detect one of 8 basic emotions</p>', unsafe_allow_html=True)

# ============================
# Model Download & Load
# ============================
MODEL_URL = "https://drive.google.com/uc?id=13RJB6HPpb_0Mx7qoPY8l-g5MzQvvU9Nd"
MODEL_PATH = "final_vgg16_affectnet.keras"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model (~160MB)... This only happens once."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.success("Model downloaded successfully!")
    st.balloons()

@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)

model = load_model()

# ============================
# Emotion Labels
# ============================
emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emoji_map = {
    'Anger': '😡', 'Contempt': '😤', 'Disgust': '🤢', 'Fear': '😨',
    'Happy': '😊', 'Neutral': '😐', 'Sad': '😢', 'Surprise': '😲'
}

# ============================
# Prediction Function
# ============================
def predict_emotion(img):
    img = img.resize((224, 224))
    arr = np.array(img)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    idx = np.argmax(preds)
    return emotions[idx], preds[idx], preds

# ============================
# Upload & Display
# ============================
uploaded_file = st.file_uploader(
    "Choose a clear face photo",
    type=["jpg", "jpeg", "png"],
    help="Best results with front-facing, well-lit portraits",
    label_visibility="collapsed"
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        if st.button("Detect Emotion", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                label, confidence, all_probs = predict_emotion(image)
                
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='emotion-big'>{emoji_map[label]}</div>", unsafe_allow_html=True)
            st.markdown(f"### **{label}**")
            st.markdown(f"#### Confidence: **{confidence:.1%}**")
            
            # Confidence bar
            st.markdown(f"""
                <div class='confidence-bar'>
                    <div class='confidence-fill' style='width: {confidence*100}%'></div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Optional detailed view
            with st.expander("View all emotion probabilities"):
                sorted_idx = np.argsort(all_probs)[::-1]
                for i in sorted_idx:
                    emo = emotions[i]
                    st.write(f"{emoji_map[emo]} **{emo}**: {all_probs[i]:.2%}")

else:
    st.info("👆 Upload a photo to get started")

# ============================
# Footer
# ============================
st.markdown("---")
st.caption("Model: VGG16 fine-tuned on AffectNet • 8 emotions • ~64% accuracy on validation set")
