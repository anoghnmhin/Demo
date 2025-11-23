# app.py
import streamlit as st
import numpy as np
import gdown
import os
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow import keras

# ============================
# CHỈNH CHỦ ĐỀ SÁNG (rất quan trọng!)
# ============================
st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="Face",
    layout="centered",
    initial_sidebar_state="collapsed",
    # BẮT BUỘC thêm dòng này để ép dùng theme sáng
    menu_items=None
)

# Ép Streamlit dùng theme sáng 100%
with open("streamlit.toml", "w") as f:
    f.write('[theme]\nprimaryColor="#FF4B4B"\nbackgroundColor="#FFFFFF"\nsecondaryBackgroundColor="#F0F2F6"\ntextColor="#31333F"\nfont="sans serif"')

# ============================
# CSS đẹp + sáng sủa
# ============================
st.markdown("""
<style>
    .main {background: #ffffff; padding: 2rem;}
    .title {font-size: 3rem; font-weight: 700; text-align: center; color: #1e293b; margin-bottom: 0.5rem;}
    .subtitle {text-align: center; color: #64748b; font-size: 1.2rem; margin-bottom: 2rem;}
    .result-card {
        background: #ffffff;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        height: 100%;
    }
    .emotion-big {font-size: 4.5rem; margin: 0.5rem 0;}
    .confidence-bar {
        height: 12px;
        background: #e2e8f0;
        border-radius: 6px;
        overflow: hidden;
        margin: 1.5rem 0;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6, #10b981);
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# Tiêu đề
# ============================
st.markdown('<h1 class="title">Face Facial Emotion Recognition</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a clear portrait photo • 8 emotions detection</p>', unsafe_allow_html=True)

# ============================
# Model
# ============================
MODEL_URL = "https://drive.google.com/uc?id=13RJB6HPpb_0Mx7qoPY8l-g5MzQvvU9Nd"
MODEL_PATH = "final_vgg16_affectnet.keras"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model (~160MB)..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.success("Model ready!")
    st.balloons()

@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)
model = load_model()

# ============================
# Emotions + emoji chuẩn
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
# Layout 2 cột – SÁNG SỦA – KHÔNG CÒN Ô TRẮNG
# ============================
uploaded_file = st.file_uploader(
    "Choose a clear face photo",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Your photo", use_column_width=True)

    with col2:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        
        if st.button("Detect Emotion", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                label, confidence, all_probs = predict_emotion(image)

            st.markdown(f"<div class='emotion-big'>{emoji_map[label]}</div>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:#1e293b'>{label}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h4>Confidence: <b>{confidence:.1%}</b></h4>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class='confidence-bar'>
                    <div class='confidence-fill' style='width: {confidence*100}%'></div>
                </div>
            """, unsafe_allow_html=True)

            with st.expander("All emotion probabilities"):
                for emo, prob in sorted(zip(emotions, all_probs), key=lambda x: x[1], reverse=True):
                    st.write(f"{emoji_map[emo]} **{emo}**: {prob:.2%}")

        else:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.info("Click **Detect Emotion** to see result")

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Upload a photo to start")

st.caption("VGG16 • Trained on AffectNet • 8 emotions")
