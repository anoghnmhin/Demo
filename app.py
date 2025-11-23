# app.py
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import io
import os
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AffectNet - Emotion Classification",
    page_icon="😊",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------- Configuration ----------
MODEL_PATH = "/mnt/data/final_vgg16_affectnet.keras"  # <-- đường dẫn file model bạn đã upload
IMG_SIZE = 224
CLASS_NAMES = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
TOP_K = 3

# ---------- Utils ----------
def load_image_from_upload(uploaded_file) -> Image.Image:
    image = Image.open(uploaded_file).convert("RGB")
    return image

def pil_to_model_input(img: Image.Image):
    """Resize, convert to numpy and run VGG16 preprocess_input"""
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img_resized).astype(np.float32)
    # VGG preprocess_input expects images in RGB and channels last
    arr = preprocess_input(arr)  # subtract mean etc.
    arr = np.expand_dims(arr, axis=0)  # batch dim
    return arr

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1, keepdims=True)

@st.cache_resource(show_spinner=False)
def load_model(path=MODEL_PATH):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        # show helpful error
        raise RuntimeError(
            f"Không thể load model từ '{path}'. Lỗi: {e}\n"
            "Hãy kiểm tra:\n"
            " - Đường dẫn model có đúng không?\n"
            " - File .keras/.h5 có tương thích với phiên bản TensorFlow hiện tại?\n"
            " - Nếu bạn deploy lên Streamlit Cloud, cân nhắc upload model lên một URL (S3/Drive) và tải xuống runtime."
        )

def predict_image(model, img: Image.Image):
    x = pil_to_model_input(img)
    preds = model.predict(x, verbose=0)
    # some models already softmax; ensure proper shape
    if preds.shape[-1] == len(CLASS_NAMES):
        probs = preds[0]
    else:
        probs = softmax(preds[0])
    # normalize just in case
    probs = probs / (probs.sum() + 1e-12)
    top_idx = probs.argsort()[::-1][:TOP_K]
    top = [(CLASS_NAMES[i], float(probs[i])) for i in top_idx]
    return top, probs

def plot_probabilities(probs):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    y_pos = np.arange(len(CLASS_NAMES))
    ax.barh(y_pos, probs, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(CLASS_NAMES)
    ax.invert_yaxis()
    ax.set_xlabel('Probability')
    ax.set_title('Emotion probabilities')
    plt.tight_layout()
    return fig

# ---------- UI ----------
st.title("🧠 AffectNet — Emotion Classification (VGG16)")
st.write(
    "Upload 1 ảnh khuôn mặt (RGB). Mô hình đã được huấn luyện trên AffectNet và dự đoán 8 lớp cảm xúc."
)

with st.sidebar:
    st.markdown("### Thông tin")
    st.write("- Input size: 224×224")
    st.write("- Classes: " + ", ".join(CLASS_NAMES))
    st.write("- Model file: `" + MODEL_PATH + "`")
    st.markdown("---")
    st.markdown("Mẹo: ảnh có khuôn mặt rõ (close-up) cho kết quả tốt hơn.")

# Load model (cached)
model_load_error = None
model = None
try:
    with st.spinner("Đang load model..."):
        model = load_model(MODEL_PATH)
except Exception as e:
    model_load_error = e

if model_load_error:
    st.error("Không thể load model. Xem thông báo lỗi trong expand below.")
    st.exception(model_load_error)
    st.stop()

# Image upload
uploaded_file = st.file_uploader("Chọn ảnh (jpg, png)", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns([1, 1])

if uploaded_file is not None:
    # Show original image
    image = load_image_from_upload(uploaded_file)
    with col1:
        st.image(image, caption="Ảnh gốc", use_column_width=True)

    # Preprocess & predict
    with st.spinner("Tiền xử lý ảnh và dự đoán..."):
        top, probs = predict_image(model, image)

    # Show result
    with col2:
        st.markdown("### 🔎 Dự đoán hàng đầu")
        for label, p in top:
            st.write(f"**{label}** — {p*100:.2f}%")
        st.markdown("---")
        st.markdown("### Chi tiết xác suất cho từng lớp")
        fig = plot_probabilities(probs)
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("### JSON kết quả (raw)")
    st.json({cls: float(p) for cls, p in zip(CLASS_NAMES, probs)})

else:
    st.info("Hãy upload một ảnh để bắt đầu (chỉ cần có khuôn mặt trong ảnh).")

st.markdown("---")
st.write("Mọi thắc mắc hoặc cần thêm tính năng (webcam, đăng ký khuôn mặt, lưu lịch sử dự đoán) thì báo mình nhé 😊")
