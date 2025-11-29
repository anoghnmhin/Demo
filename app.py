import os
# --- FIX LỖI KERAS 3 VS TRANSFORMERS ---
# Phải đặt biến này TRƯỚC KHI import tensorflow/keras
os.environ["TF_USE_LEGACY_KERAS"] = "0"  # Dùng Keras 3 native
os.environ["KERAS_BACKEND"] = "tensorflow"

import streamlit as st
import sys
import traceback
import numpy as np
import pickle
import pandas as pd
from PIL import Image

# Cấu hình trang
st.set_page_config(page_title="Flower Classifier AI", page_icon="🌸", layout="wide")

try:
    import tensorflow as tf
    from tensorflow import keras
    from transformers import TFViTModel
except ImportError as e:
    st.error(f"❌ Thiếu thư viện: {e}")
    st.info("Hãy đảm bảo requirements.txt có: tensorflow-cpu, transformers, tf-keras")
    st.stop()

# ------------------------------------------------------------------
# 1. ĐỊNH NGHĨA CUSTOM LAYER (ViT)
# ------------------------------------------------------------------
@keras.saving.register_keras_serializable()
class ViTFeatureExtractorLayer(keras.layers.Layer):
    def __init__(self, model_name='google/vit-base-patch16-224', **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        # Load model từ Hugging Face
        self.vit = TFViTModel.from_pretrained(self.model_name)

    def call(self, inputs):
        outputs = self.vit(inputs)
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
FLOWER_EMOJIS = {'Daisy': '🌼', 'Dandelion': '🏵️', 'Rose': '🌹', 'Sunflower': '🌻', 'Tulip': '🌷'}

@st.cache_resource
def load_models():
    # Kiểm tra file tồn tại
    if not os.path.exists(PATH_EXTRACTOR): return None, None, None, f"Thiếu file {PATH_EXTRACTOR}"
    if not os.path.exists(PATH_SCALER): return None, None, None, f"Thiếu file {PATH_SCALER}"
    if not os.path.exists(PATH_CLASSIFIER_WEIGHTS): return None, None, None, f"Thiếu file {PATH_CLASSIFIER_WEIGHTS}"

    try:
        # A. Load Extractor
        extractor = keras.models.load_model(PATH_EXTRACTOR)
        
        # B. Load Scaler
        with open(PATH_SCALER, 'rb') as f:
            scaler = pickle.load(f)
            
        # C. Load Classifier
        # Input shape 768 là chuẩn output của ViT Base
        classifier = keras.Sequential([
            keras.layers.InputLayer(input_shape=(768,)),
            keras.layers.Dense(NUM_CLASSES, activation='softmax')
        ])
        classifier.load_weights(PATH_CLASSIFIER_WEIGHTS)
        
        return extractor, scaler, classifier, None
    except Exception:
        return None, None, None, traceback.format_exc()

# Load models
extractor, scaler, classifier, err = load_models()

# ------------------------------------------------------------------
# 3. GIAO DIỆN & DỰ ĐOÁN
# ------------------------------------------------------------------
if err:
    st.error("🚨 Lỗi khởi động Model:")
    st.code(err)
    st.stop()

st.title("🌸 Phân Loại Hoa (ViT + Keras 3)")
st.caption("Hệ thống sử dụng Vision Transformer và Transfer Learning")

uploaded_file = st.file_uploader("Chọn ảnh hoa...", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.image(image, use_column_width=True)
        
    if st.button("🔍 Phân tích", type="primary"):
        with st.spinner("Đang xử lý..."):
            try:
                # Pipeline xử lý
                img_resized = image.resize(IMG_SIZE)
                img_array = np.array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) # (1, 224, 224, 3)

                # Predict
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
