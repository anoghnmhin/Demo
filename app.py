import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os
from PIL import Image
from tensorflow import keras

# ------------------------------------------------------------------
# 1. CẤU HÌNH TRANG (PAGE CONFIG)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Flower Classifier AI",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------
# 2. CẤU HÌNH ĐƯỜNG DẪN FILE (PATH CONFIG)
# ------------------------------------------------------------------
# Lấy đường dẫn tuyệt đối của thư mục chứa file app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Định nghĩa đường dẫn tới các file model
# Lưu ý: Tên file phải chính xác 100% với file bạn đang có trong thư mục
PATH_EXTRACTOR = os.path.join(BASE_DIR, "vit_transfer_feature_extractor.keras")
PATH_SCALER = os.path.join(BASE_DIR, "feature_scaler (1).pkl")
PATH_CLASSIFIER_WEIGHTS = os.path.join(BASE_DIR, "vit_transfer_model.weights.h5")

# Thông số kỹ thuật (Phải khớp với lúc Train model)
IMG_SIZE = (224, 224)   # Kích thước ảnh đầu vào cho ViT
FEATURE_DIM = 768       # Số chiều đặc trưng của ViT Base
NUM_CLASSES = 5         # Số lượng loài hoa (Daisy, Dandelion, Rose, Sunflower, Tulip)

# Danh sách nhãn (Labels)
FLOWER_CLASSES = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']
FLOWER_EMOJIS = {'Daisy': '🌼', 'Dandelion': '🏵️', 'Rose': '🌹', 'Sunflower': '🌻', 'Tulip': '🌷'}

# ------------------------------------------------------------------
# 3. HÀM LOAD MODEL (CACHED)
# ------------------------------------------------------------------
@st.cache_resource
def load_system_components():
    """
    Load toàn bộ 3 thành phần: Extractor, Scaler, Classifier
    Dùng cache để không phải load lại mỗi lần bấm nút.
    """
    # Kiểm tra file tồn tại
    if not os.path.exists(PATH_EXTRACTOR):
        st.error(f"❌ Không tìm thấy file: {PATH_EXTRACTOR}")
        return None, None, None
    if not os.path.exists(PATH_SCALER):
        st.error(f"❌ Không tìm thấy file: {PATH_SCALER}")
        return None, None, None
    if not os.path.exists(PATH_CLASSIFIER_WEIGHTS):
        st.error(f"❌ Không tìm thấy file: {PATH_CLASSIFIER_WEIGHTS}")
        return None, None, None

    try:
        # A. Load ViT Feature Extractor
        extractor = keras.models.load_model(PATH_EXTRACTOR)
        print("✅ Loaded Feature Extractor")

        # B. Load Scaler
        with open(PATH_SCALER, 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Loaded Scaler")

        # C. Build & Load Classifier
        # Dựng lại khung sườn (Architecture) cho Classifier
        classifier = keras.Sequential([
            keras.layers.InputLayer(input_shape=(FEATURE_DIM,)),
            # Nếu lúc train bạn có Dropout, hãy uncomment dòng dưới:
            # keras.layers.Dropout(0.2), 
            keras.layers.Dense(NUM_CLASSES, activation='softmax')
        ])
        classifier.load_weights(PATH_CLASSIFIER_WEIGHTS)
        print("✅ Loaded Classifier Weights")

        return extractor, scaler, classifier

    except Exception as e:
        st.error(f"❌ Lỗi nghiêm trọng khi load model: {str(e)}")
        return None, None, None

# Gọi hàm load ngay khi app khởi động
extractor, scaler, classifier = load_system_components()

# ------------------------------------------------------------------
# 4. HÀM DỰ ĐOÁN (PREDICTION PIPELINE)
# ------------------------------------------------------------------
def predict_flower(img_pil):
    """
    Quy trình: Ảnh -> Resize -> ViT Extract -> Scale -> Classify
    """
    # 1. Tiền xử lý ảnh
    img = img_pil.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

    # 2. Trích xuất đặc trưng (Feature Extraction)
    features = extractor.predict(img_array, verbose=0) # Shape: (1, 768)

    # 3. Chuẩn hóa dữ liệu (Standard Scaling)
    features_scaled = scaler.transform(features)

    # 4. Phân loại (Classification)
    preds = classifier.predict(features_scaled, verbose=0)[0] # Trả về mảng xác suất

    # 5. Lấy kết quả cao nhất
    max_index = np.argmax(preds)
    confidence = preds[max_index]
    label = FLOWER_CLASSES[max_index]

    return label, confidence, preds

# ------------------------------------------------------------------
# 5. GIAO DIỆN NGƯỜI DÙNG (UI/UX)
# ------------------------------------------------------------------

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1822/1822167.png", width=100)
    st.title("Trợ lý AI Thực Vật")
    st.markdown("---")
    st.info(
        """
        **Mô hình:** Vision Transformer (ViT)
        **Phương pháp:** Transfer Learning + SVM Architecture
        **Độ chính xác:** Cao trên tập dữ liệu hoa chuẩn.
        """
    )
    st.markdown("### Các loài hoa hỗ trợ:")
    for flower in FLOWER_CLASSES:
        st.write(f"- {FLOWER_EMOJIS[flower]} {flower}")

# --- Main Content ---
st.title("🌸 Nhận Diện Loài Hoa Bằng AI")
st.markdown("##### Tải ảnh bông hoa lên để hệ thống phân tích...")

# Widget upload file
uploaded_file = st.file_uploader("Chọn ảnh (jpg, png, jpeg)...", type=["jpg", "jpeg", "png", "webp"])

# Chỉ hiển thị giao diện phân tích khi đã load model thành công
if extractor and scaler and classifier:
    if uploaded_file is not None:
        # Chia cột: Bên trái ảnh, Bên phải kết quả
        col1, col2 = st.columns([1, 1.2], gap="large")

        try:
            image = Image.open(uploaded_file).convert("RGB")
            
            with col1:
                st.image(image, caption="Ảnh bạn tải lên", use_column_width=True)

            # Nút bấm dự đoán
            if st.button("🔍 Phân tích ngay", use_container_width=True, type="primary"):
                with st.spinner("Đang trích xuất đặc trưng qua mạng Neural..."):
                    
                    label, conf, all_probs = predict_flower(image)

                # Hiển thị kết quả bên phải
                with col2:
                    st.success("Đã phân tích xong!")
                    
                    # Hiển thị tên hoa to và đẹp
                    emoji = FLOWER_EMOJIS.get(label, '🌸')
                    st.markdown(f"<h2 style='text-align: center; color: #FF4B4B;'>{emoji} {label}</h2>", unsafe_allow_html=True)
                    
                    # Thanh đo độ tin cậy
                    st.metric("Độ chính xác", f"{conf:.1%}")
                    st.progress(float(conf))

                    st.markdown("---")
                    
                    # Biểu đồ chi tiết
                    st.write("**Tỷ lệ dự đoán chi tiết:**")
                    df_probs = pd.DataFrame({
                        'Loài hoa': FLOWER_CLASSES,
                        'Tỷ lệ': all_probs
                    })
                    st.bar_chart(df_probs.set_index('Loài hoa'), color="#FF69B4")

        except Exception as e:
            st.error(f"Có lỗi khi xử lý ảnh: {e}")
else:
    st.warning("⚠️ Hệ thống đang khởi động hoặc thiếu file model. Vui lòng kiểm tra lại thư mục deploy.")
