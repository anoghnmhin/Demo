import streamlit as st
import numpy as np
import gdown
import os
from PIL import Image
import time
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow import keras

# ============================
# CÀI ĐẶT TRANG
# ============================
st.set_page_config(
    page_title="Nhận diện cảm xúc khuôn mặt",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS để làm đẹp giao diện
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
    }
    .title {
        font-size: 3rem !important;
        font-weight: 800;
        text-align: center;
        background: -webkit-linear-gradient(45deg, #f5f7fa, #c3cfe2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #e0e0e0;
        margin-bottom: 2rem;
    }
    .upload-box {
        border: 3px dashed #ffffff50;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: rgba(255,255,255,0.1);
        transition: all 0.3s;
    }
    .upload-box:hover {
        border-color: #ffffff90;
        background: rgba(255,255,255,0.2);
    }
    .result-box {
        background: rgba(255,255,255,0.15);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .emotion-emoji {
        font-size: 5rem;
        margin: 1rem 0;
    }
    .confidence-bar {
        height: 20px;
        background: rgba(255,255,255,0.2);
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# TIÊU ĐỀ & GIỚI THIỆU
# ============================
st.markdown('<h1 class="title">🎭 Emotion Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Phát hiện 8 cảm xúc cơ bản từ khuôn mặt: giận dữ, khinh thường, ghê tởm, sợ hãi, vui vẻ, trung lập, buồn, ngạc nhiên</p>', unsafe_allow_html=True)

# ============================
# DOWNLOAD MODEL
# ============================
MODEL_URL = "https://drive.google.com/uc?id=13RJB6HPpb_0Mx7qoPY8l-g5MzQvvU9Nd"
MODEL_PATH = "final_vgg16_affectnet.keras"

def download_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("🔽 Đang tải mô hình VGG16-AffectNet (~160MB), lần đầu sẽ mất chút thời gian...")
        progress_bar = st.progress(0)
        with st.spinner("Đang tải mô hình..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
        st.success("✅ Tải mô hình thành công!")
        st.balloons()

with st.spinner("Kiểm tra mô hình..."):
    download_model()

# ============================
# LOAD MODEL
# ============================
@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)

model = load_model()

# ============================
# DANH SÁCH CẢM XÚC + EMOJI
# ============================
emotion_classes = [
    'anger', 'contempt', 'disgust', 'fear',
    'happy', 'neutral', 'sad', 'surprise'
]

emoji_map = {
    'anger': '😡', 'contempt': '😤', 'disgust': '🤢', 'fear': '😨',
    'happy': '😊', 'neutral': '😐', 'sad': '😢', 'surprise': '😲'
}

vietnamese_names = {
    'anger': 'Giận dữ', 'contempt': 'Khinh thường', 'disgust': 'Ghê tởm',
    'fear': 'Sợ hãi', 'happy': 'Vui vẻ', 'neutral': 'Trung lập',
    'sad': 'Buồn bã', 'surprise': 'Ngạc nhiên'
}

# ============================
# HÀM DỰ ĐOÁN
# ============================
def predict_emotion(img):
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array, verbose=0)[0]
    label_idx = np.argmax(preds)
    confidence = preds[label_idx]
    label = emotion_classes[label_idx]
    return label, confidence, preds

# ============================
# UPLOAD & HIỂN THỊ
# ============================
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "📸 Upload ảnh khuôn mặt của bạn",
        type=["jpg", "jpeg", "png"],
        help="Chọn ảnh rõ mặt, không che khuất"
    )
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    with col2:
        st.image(image, caption="Ảnh của bạn", use_column_width=True, clamp=True)

    if st.button("🔮 Phân tích cảm xúc ngay!", type="primary", use_container_width=True):
        with st.spinner("Đang phân tích cảm xúc..."):
            time.sleep(1.5)  # Tạo hiệu ứng mượt
            label, confidence, all_preds = predict_emotion(image)
            
            st.markdown("---")
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            
            # Kết quả chính
            st.markdown(f"<h1 class='emotion-emoji'>{emoji_map[label]}</h1>", unsafe_allow_html=True)
            st.markdown(f"### **{vietnamese_names[label].upper()}**")
            st.markdown(f"#### Độ tin cậy: **{confidence:.1%}**")
            
            # Thanh độ tin cậy
            st.markdown(f"<div class='confidence-bar'><div style='width: {confidence*100:.1f}%; height:100%; background: linear-gradient(90deg, #ff6b6b, #4ecdc4); border-radius: 10px;'></div></div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Top 3 cảm xúc (tùy chọn mở rộng)
            with st.expander("📊 Xem chi tiết tất cả cảm xúc"):
                sorted_idx = np.argsort(all_preds)[::-1]
                for i in sorted_idx[:5]:
                    emo = emotion_classes[i]
                    st.write(f"{emoji_map[emo]} **{vietnamese_names[emo]}**: {all_preds[i]:.2%}")

else:
    with col2:
        st.info("👈 Hãy upload một bức ảnh để bắt đầu phân tích cảm xúc!")

# ============================
# FOOTER
# ============================
st.markdown("---")
st.caption("🚀 Được huấn luyện trên tập dữ liệu AffectNet • Mô hình VGG16 • Độ chính xác ~64% trên tập kiểm tra")
