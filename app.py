# app.py
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
    page_icon="Face",
    layout="centered"
)

# ============================
# Light & Clean CSS (giữ phong cách cũ nhưng đẹp hơn)
# ============================
st.markdown("""
<style>
    .main {background: #f8fafc; padding: 2rem;}
    .title {font-size: 3rem; font-weight: 700; text-align: center; color: #1e293b;}
    .subtitle {text-align: center; color: #64748b; font-size: 1.2rem; margin-bottom: 2rem;}
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
        height: 100%;
    }
    .emotion-big {font-size: 4.5rem; margin: 0.5rem 0;}
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
    }
</style>
""", unsafe_allow_html=True)

# ============================
# Title
# ============================
st.markdown('<h1 class="title">Face Facial Emotion Recognition</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a clear portrait to detect one of 8 basic emotions</p>', unsafe_allow_html=True)

# ============================
# Model
# ============================
MODEL_URL = "https://drive.google.com/uc?id=13RJB6HPpb_0Mx7qoPY8l-g5MzQvvU9Nd"
MODEL_PATH = "final_vgg16_affectnet.keras"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model (~160MB)..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.success("Model downloaded!")
    st.balloons()

@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)
model = load_model()

# ============================
# Emotions
# ============================
emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emoji_map = {
    'Anger': 'Angry', 'Contempt': 'Smirking', 'Disgust': 'Disgusted', 'Fear': 'Fearful',
    'Happy': 'Grinning', 'Neutral': 'Neutral', 'Sad': 'Crying', 'Surprise': 'Shocked'
}

def predict_emotion(img):
    img = img.resize((224, 224))
    arr = np.array(img)
    arr = preprocess_input(arr.astype(np.float32))
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    idx = np.argmax(preds)
    return emotions[idx], preds[idx], preds

# ============================
# Main Layout: 2 cột như cũ, KHÔNG CÒN Ô TRẮNG THỪA
# ============================
uploaded_file = st.file_uploader(
    "Choose a clear face photo",
    type=["jpg", "jpeg", "png"],
    help="Front-facing and well-lit photos work best",
    label_visibility="collapsed"
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        
        if st.button("Detect Emotion", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                label, confidence, all_probs = predict_emotion(image)

            # Kết quả hiện ngay trong cột phải
            st.markdown(f"<div class='emotion-big'>{emoji_map[label]}</div>", unsafe_allow_html=True)
            st.markdown(f"### **{label}**", unsafe_allow_html=True)
            st.markdown(f"#### Confidence: **{confidence:.1%}**")
            
            st.markdown(f"""
                <div class='confidence-bar'>
                    <div class='confidence-fill' style='width: {confidence*100}%'></div>
                </div>
            """, unsafe_allow_html=True)

            with st.expander("View all probabilities"):
                sorted_idx = np.argsort(all_probs)[::-1]
                for i in sorted_idx:
                    emo = emotions[i]
                    st.write(f"{emoji_map[emo]} **{emo}**: {all_probs[i]:.2%}")
        else:
            # Khi chưa bấm nút → hiện placeholder đẹp thay vì ô trắng
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.info("👈 Click **Detect Emotion** to see the result")

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Upload a photo to begin")

# ============================
# Footer
# ============================
st.markdown("---")
st.caption("VGG16 • AffectNet Dataset • 8 Emotions • Made with Streamlit")
