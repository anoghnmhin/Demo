import streamlit as st
import numpy as np
import gdown
import os
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow import keras

# ----------------------------
# 1. Link Google Drive (Direct)
# ----------------------------
MODEL_URL = "https://drive.google.com/uc?id=13RJB6HPpb_0Mx7qoPY8l-g5MzQvvU9Nd"
MODEL_PATH = "final_vgg16_affectnet.keras"

# ----------------------------
# 2. Tải model nếu chưa tồn tại
# ----------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Đang tải mô hình, vui lòng chờ ~160MB..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.success("Tải mô hình thành công!")

download_model()

# ----------------------------
# 3. Load model (cache để không load lại)
# ----------------------------
@st.cache_resource
def load_model():
    model = keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ----------------------------
# 4. Danh sách cảm xúc
# ----------------------------
emotion_classes = [
    'anger', 'contempt', 'disgust', 'fear',
    'happy', 'neutral', 'sad', 'surprise'
]

# ----------------------------
# 5. Hàm dự đoán cảm xúc
# ----------------------------
def predict_emotion(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]
    label_index = np.argmax(preds)
    confidence = preds[label_index]

    return emotion_classes[label_index], confidence

# ----------------------------
# 6. Giao diện Streamlit
# ----------------------------
st.title("🎭 Nhận Diện Cảm Xúc Khuôn Mặt (AffectNet + VGG16)")
st.write("Upload một ảnh chân dung để dự đoán cảm xúc.")

uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Ảnh bạn đã upload", use_column_width=True)

    if st.button("Dự đoán"):
        label, conf = predict_emotion(image)

        st.subheader(f"🔍 Kết quả: **{label.upper()}**")
        st.write(f"Độ tin cậy: **{conf:.2f}**")

