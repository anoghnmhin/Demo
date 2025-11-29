import streamlit as st
import numpy as np
import pickle
import pandas as pd
from PIL import Image
from tensorflow import keras

# ----------------------------
# 0. Cấu hình trang (Theme Hoa)
# ----------------------------
st.set_page_config(
    page_title="Flower Classifier",
    page_icon="🌸",
    layout="wide"
)

# ----------------------------
# 1. Cấu hình File & Model
# ----------------------------
# Tên file giữ nguyên như bạn đã upload
PATH_EXTRACTOR = "vit_transfer_feature_extractor.keras"
PATH_SCALER = "feature_scaler (1).pkl"
PATH_CLASSIFIER_WEIGHTS = "vit_transfer_model.weights.h5"

# CẤU HÌNH LẠI CHO BÀI TOÁN HOA
# Ví dụ: Dataset hoa thường có 5 loại (Daisy, Dandelion, Rose, Sunflower, Tulip)
# Bạn hãy sửa số này cho khớp với lúc train
NUM_CLASSES = 5  
IMG_SIZE = (224, 224)
FEATURE_DIM = 768 

# ----------------------------
# 2. Load Pipeline
# ----------------------------
@st.cache_resource
def load_components():
    # 1. Extractor
    try:
        feature_extractor = keras.models.load_model(PATH_EXTRACTOR)
    except Exception as e:
        st.error(f"Lỗi load Extractor: {e}")
        return None, None, None

    # 2. Scaler
    try:
        with open(PATH_SCALER, 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        st.error(f"Lỗi load Scaler: {e}")
        return None, None, None

    # 3. Classifier (Head)
    try:
        # Dựng lại kiến trúc lớp cuối (Output layer)
        # Lưu ý: Nếu bài toán hoa có 5 lớp, Dense phải là 5
        classifier = keras.Sequential([
            keras.layers.InputLayer(input_shape=(FEATURE_DIM,)),
            # Nếu lúc train có Dropout hay Dense ẩn, thêm vào đây
            # keras.layers.Dropout(0.2), 
            keras.layers.Dense(NUM_CLASSES, activation='softmax')
        ])
        classifier.load_weights(PATH_CLASSIFIER_WEIGHTS)
    except Exception as e:
        st.error(f"Lỗi load Classifier (Sai kiến trúc hoặc file hỏng): {e}")
        return None, None, None

    return feature_extractor, scaler, classifier

extractor, scaler, classifier = load_components()

# ----------------------------
# 3. Định nghĩa Nhãn Hoa
# ----------------------------
# SỬA LẠI DANH SÁCH NÀY THEO ĐÚNG THỨ TỰ LÚC TRAIN
flower_classes = [
    'Daisy',      # Hoa cúc dại
    'Dandelion',  # Bồ công anh
    'Rose',       # Hoa hồng
    'Sunflower',  # Hướng dương
    'Tulip'       # Tulip
]

flower_emojis = {
    'Daisy': '🌼',
    'Dandelion': '🏵️',
    'Rose': '🌹',
    'Sunflower': '🌻',
    'Tulip': '🌷'
}

# ----------------------------
# 4. Hàm Dự đoán
# ----------------------------
def predict_flower(img_pil):
    # Preprocess
    img = img_pil.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) # (1, 224, 224, 3)

    # 1. Trích xuất đặc trưng
    features = extractor.predict(img_array, verbose=0)
    
    # 2. Chuẩn hóa (Scaler)
    features_scaled = scaler.transform(features)

    # 3. Phân loại
    preds = classifier.predict(features_scaled, verbose=0)[0]
    
    idx = np.argmax(preds)
    conf = preds[idx]
    
    return flower_classes[idx], conf, preds

# ----------------------------
# 5. Giao diện (UI)
# ----------------------------
with st.sidebar:
    st.title("🌿 Vườn Thực Vật AI")
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=100)
    st.info(
        """
        Ứng dụng sử dụng **Vision Transformer (ViT)** để trích xuất đặc trưng hoa.
        
        **Các loài hoa hỗ trợ:**
        - 🌼 Daisy
        - 🏵️ Dandelion
        - 🌹 Rose
        - 🌻 Sunflower
        - 🌷 Tulip
        """
    )

st.title("🌸 Nhận Diện Loài Hoa (Flower Classification)")
st.markdown("### Tải ảnh hoa lên để AI định danh")

uploaded_file = st.file_uploader("Chọn ảnh hoa...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file and extractor and scaler and classifier:
    col1, col2 = st.columns([1, 1])
    
    image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.image(image, caption="Ảnh bạn đã tải lên", use_container_width=True)

    if st.button("🔍 Định danh loài hoa", use_container_width=True):
        with st.spinner("Đang quan sát cánh hoa..."):
            try:
                label, conf, all_probs = predict_flower(image)
                
                with col2:
                    st.success("Đã có kết quả!")
                    
                    # Hiển thị kết quả to đẹp
                    emoji = flower_emojis.get(label, '🌸')
                    st.metric(
                        label="Đây có thể là:",
                        value=f"{emoji} {label}",
                        delta=f"Độ tin cậy: {conf:.1%}"
                    )
                    
                    st.progress(float(conf))
                
                # Biểu đồ xác suất bên dưới
                st.divider()
                st.subheader("📊 Phân tích chi tiết")
                
                df_probs = pd.DataFrame({
                    'Loài hoa': flower_classes,
                    'Tỷ lệ': all_probs
                })
                
                # Tô màu cột cao nhất
                st.bar_chart(df_probs.set_index('Loài hoa'), color="#FF69B4") # Màu hồng
                
            except Exception as e:
                st.error(f"Có lỗi khi dự đoán: {e}")
                st.warning("Gợi ý: Kiểm tra lại xem số lượng lớp (NUM_CLASSES) trong code có khớp với file weights.h5 không.")
