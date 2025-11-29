import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# --- QUAN TRỌNG: Import thư viện Hugging Face ---
try:
    from transformers import TFViTModel
except ImportError:
    st.error("⚠️ Thiếu thư viện 'transformers'. Vui lòng thêm vào requirements.txt")
    st.stop()

# ------------------------------------------------------------------
# 1. ĐỊNH NGHĨA CUSTOM LAYER (FIX LỖI SERIALIZATION)
# ------------------------------------------------------------------
# Đây là đoạn code bị thiếu khiến Keras không load được model
@keras.saving.register_keras_serializable()
class ViTFeatureExtractorLayer(keras.layers.Layer):
    def __init__(self, model_name='google/vit-base-patch16-224', **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        # Load lõi ViT từ Hugging Face
        self.vit = TFViTModel.from_pretrained(self.model_name)

    def call(self, inputs):
        # inputs shape: (batch_size, 3, 224, 224) hoặc (batch, 224, 224, 3) tùy config
        # TFViTModel trả về TFBaseModelOutputWithPooling
        outputs = self.vit(inputs)
        
        # Lấy CLS token (vector đặc trưng đầu tiên đại diện cho cả ảnh)
        # Shape output: (batch_size, 768)
        return outputs.last_hidden_state[:, 0, :]

    def get_config(self):
        config = super().get_config()
        config.update({"model_name": self.model_name})
        return config

# ------------------------------------------------------------------
# 2. CẤU HÌNH TRANG & ĐƯỜNG DẪN
# ------------------------------------------------------------------
st.set_page_config(page_title="Flower Classifier AI", page_icon="🌸", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_EXTRACTOR = os.path.join(BASE_DIR, "vit_transfer_feature_extractor.keras")
PATH_SCALER = os.path.join(BASE_DIR, "feature_scaler (1).pkl")
PATH_CLASSIFIER_WEIGHTS = os.path.join(BASE_DIR, "vit_transfer_model.weights.h5")

# Thông số (Khớp với config lỗi: input shape channels first [3, 224, 224])
IMG_SIZE = (224, 224)
NUM_CLASSES = 5
FLOWER_CLASSES = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']
FLOWER_EMOJIS = {'Daisy': '🌼', 'Dandelion': '🏵️', 'Rose': '🌹', 'Sunflower': '🌻', 'Tulip': '🌷'}

# ------------------------------------------------------------------
# 3. HÀM LOAD MODEL
# ------------------------------------------------------------------
@st.cache_resource
def load_system_components():
    # Kiểm tra file
    if not os.path.exists(PATH_EXTRACTOR): return None, None, None, f"Thiếu file {PATH_EXTRACTOR}"
    if not os.path.exists(PATH_SCALER): return None, None, None, f"Thiếu file {PATH_SCALER}"
    if not os.path.exists(PATH_CLASSIFIER_WEIGHTS): return None, None, None, f"Thiếu file {PATH_CLASSIFIER_WEIGHTS}"

    try:
        # A. Load ViT Extractor (Kèm Custom Object)
        # Vì đã dùng decorator @register_keras_serializable, ta có thể load thẳng
        extractor = keras.models.load_model(PATH_EXTRACTOR)
        print("✅ Loaded Feature Extractor")

        # B. Load Scaler
        with open(PATH_SCALER, 'rb') as f:
            scaler = pickle.load(f)
        
        # C. Load Classifier
        # Lưu ý: Model Feature Extractor trả về vector (768,)
        classifier = keras.Sequential([
            keras.layers.InputLayer(input_shape=(768,)),
            keras.layers.Dense(NUM_CLASSES, activation='softmax')
        ])
        classifier.load_weights(PATH_CLASSIFIER_WEIGHTS)
        
        return extractor, scaler, classifier, None

    except Exception as e:
        return None, None, None, str(e)

extractor, scaler, classifier, error_msg = load_system_components()

# ------------------------------------------------------------------
# 4. LOGIC DỰ ĐOÁN
# ------------------------------------------------------------------
def predict_flower(img_pil):
    # 1. Resize
    img = img_pil.resize(IMG_SIZE)
    img_array = np.array(img)
    
    # 2. Xử lý kênh màu (Channels)
    # Lỗi log cho thấy input_shape model là [None, 3, 224, 224] (Channels First - NCHW)
    # Nhưng ảnh PIL/Numpy mặc định là (224, 224, 3) (Channels Last - NHWC)
    
    # Ta cần transpose nếu model yêu cầu channels first
    # Tuy nhiên, HuggingFace TFViT thường tự handle hoặc cần check input layer
    # Dựa vào log lỗi "keras_history: ['permute', 0, 0]", có thể model đã có lớp Permute.
    # Ta cứ đưa vào (1, 224, 224, 3), nếu model có lớp Permute đầu tiên nó sẽ tự xoay.
    
    img_array = np.expand_dims(img_array, axis=0) # (1, 224, 224, 3)

    # 3. Predict
    features = extractor.predict(img_array, verbose=0)
    features_scaled = scaler.transform(features)
    preds = classifier.predict(features_scaled, verbose=0)[0]
    
    idx = np.argmax(preds)
    return FLOWER_CLASSES[idx], preds[idx], preds

# ------------------------------------------------------------------
# 5. GIAO DIỆN
# ------------------------------------------------------------------
with st.sidebar:
    st.title("🌺 Flower AI")
    st.info("Sửa lỗi: Custom Layer deserialization & Transformers dependency.")

st.title("🌸 Phân Loại Hoa (ViT Patch16)")

if error_msg:
    st.error(f"❌ Lỗi khởi động: {error_msg}")
    st.stop()

uploaded_file = st.file_uploader("Upload ảnh hoa...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file).convert("RGB")
    with col1:
        st.image(image, use_column_width=True)
    
    if st.button("Phân tích", type="primary"):
        with st.spinner("Đang chạy mô hình ViT..."):
            try:
                label, conf, all_probs = predict_flower(image)
                with col2:
                    st.success(f"Kết quả: {label}")
                    st.metric("Độ tin cậy", f"{conf:.1%}")
                    
                    df = pd.DataFrame({'Hoa': FLOWER_CLASSES, 'Tỷ lệ': all_probs})
                    st.bar_chart(df.set_index('Hoa'))
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {e}")
