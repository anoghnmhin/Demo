import streamlit as st
import os
import sys
import traceback
import numpy as np
import pickle
import pandas as pd
from PIL import Image

# ------------------------------------------------------------------
# 0. CẤU HÌNH MÔI TRƯỜNG (Đặt ngay đầu file)
# ------------------------------------------------------------------
# Ép buộc sử dụng Keras 3 và Backend TensorFlow
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras  # <--- Dùng Keras 3 Standalone (QUAN TRỌNG)
import tensorflow as tf # Dùng riêng TF để xử lý tensor nếu cần

# Cấu hình trang
st.set_page_config(page_title="Flower Classifier AI", page_icon="🌸", layout="wide")

# Kiểm tra thư viện Transformers
try:
    from transformers import TFViTModel
except ImportError:
    st.error("❌ Thiếu thư viện 'transformers'.")
    st.stop()

# ------------------------------------------------------------------
# 1. ĐỊNH NGHĨA CUSTOM LAYER (Chuẩn Keras 3)
# ------------------------------------------------------------------
@keras.saving.register_keras_serializable()
class ViTFeatureExtractorLayer(keras.layers.Layer):
    def __init__(self, model_name='google/vit-base-patch16-224', **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        # Khởi tạo model HuggingFace
        # Lưu ý: Transformers trả về TF Keras Model (Legacy), 
        # nhưng Keras 3 có thể wrap được nó.
        self.vit = TFViTModel.from_pretrained(self.model_name)

    def call(self, inputs):
        # inputs shape: (batch, 3, 224, 224) hoặc (batch, 224, 224, 3)
        outputs = self.vit(inputs)
        # Lấy CLS token
        return outputs.last_hidden_state[:, 0, :]

    def get_config(self):
        config = super().get_config()
        config.update({"model_name": self.model_name})
        return config

# ------------------------------------------------------------------
# 2. LOAD SYSTEM
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_EXTRACTOR = os.path.join(BASE_DIR, "vit_transfer_feature_extractor.keras")
PATH_SCALER = os.path.join(BASE_DIR, "feature_scaler (1).pkl")
PATH_CLASSIFIER_WEIGHTS = os.path.join(BASE_DIR, "vit_transfer_model.weights.h5")

# Thông số kỹ thuật
IMG_SIZE = (224, 224)
NUM_CLASSES = 5
FLOWER_CLASSES = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

@st.cache_resource
def load_models():
    # Kiểm tra file
    if not os.path.exists(PATH_EXTRACTOR): return None, None, None, f"Thiếu file {PATH_EXTRACTOR}"
    if not os.path.exists(PATH_SCALER): return None, None, None, f"Thiếu file {PATH_SCALER}"
    if not os.path.exists(PATH_CLASSIFIER_WEIGHTS): return None, None, None, f"Thiếu file {PATH_CLASSIFIER_WEIGHTS}"

    try:
        # A. Load Extractor bằng Keras 3
        # safe_mode=False để cho phép load custom layer phức tạp
        extractor = keras.models.load_model(PATH_EXTRACTOR, safe_mode=False)
        print("✅ Extractor Loaded (Keras 3)")
        
        # B. Load Scaler
        with open(PATH_SCALER, 'rb') as f:
            scaler = pickle.load(f)
            
        # C. Load Classifier
        # Dựng lại architecture bằng Keras 3
        classifier = keras.Sequential([
            keras.layers.InputLayer(input_shape=(768,)),
            keras.layers.Dense(NUM_CLASSES, activation='softmax')
        ])
        classifier.load_weights(PATH_CLASSIFIER_WEIGHTS)
        print("✅ Classifier Loaded")
        
        return extractor, scaler, classifier, None
    except Exception:
        return None, None, None, traceback.format_exc()

# Gọi hàm load
extractor, scaler, classifier, err = load_models()

# ------------------------------------------------------------------
# 3. GIAO DIỆN & DỰ ĐOÁN
# ------------------------------------------------------------------
if err:
    st.error("🚨 Lỗi khởi động Model:")
    st.code(err)
    st.warning("Gợi ý: Nếu lỗi liên quan đến 'tf_keras', hãy xóa 'tf-keras' khỏi requirements.txt và Re-deploy.")
    st.stop()

st.title("🌸 Phân Loại Hoa (Keras 3 Native)")

uploaded_file = st.file_uploader("Chọn ảnh hoa...", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.image(image, use_column_width=True)
        
    if st.button("🔍 Phân tích", type="primary"):
        with st.spinner("Đang xử lý..."):
            try:
                # Resize
                img_resized = image.resize(IMG_SIZE)
                img_array = np.array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) 

                # Predict
                # Do dùng Keras 3, ta chuyển input thành Tensor chuẩn
                features = extractor.predict(img_array, verbose=0)
                features_scaled = scaler.transform(features)
                preds = classifier.predict(features_scaled, verbose=0)[0]
                
                # Kết quả
                idx = np.argmax(preds)
                label = FLOWER_CLASSES[idx]
                conf = preds[idx]

                with col2:
                    st.success(f"Kết quả: **{label}**")
                    st.metric("Độ tin cậy", f"{conf:.1%}")
                    
                    df = pd.DataFrame({'Loài hoa': FLOWER_CLASSES, 'Tỷ lệ': preds})
                    st.bar_chart(df.set_index('Loài hoa'))

            except Exception as e:
                st.error("Lỗi khi dự đoán:")
                st.code(traceback.format_exc())
