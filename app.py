import streamlit as st
import numpy as np
import pickle
import os
from PIL import Image
import tensorflow as tf

# ====================== CONFIG ======================
st.set_page_config(page_title="Flower Classifier AI", page_icon="Flowers", layout="wide", initial_sidebar_state="expanded")

BASE_DIR = os.path.dirname(__file__)
PATH_EXTRACTOR = os.path.join(BASE_DIR, "vit_transfer_feature_extractor.keras")
PATH_SCALER    = os.path.join(BASE_DIR, "feature_scaler (1).pkl")
PATH_WEIGHTS   = os.path.join(BASE_DIR, "vit_transfer_model.weights.h5")

IMG_SIZE = (224, 224)
CLASSES = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']
EMOJIS = {'Daisy': 'White flower', 'Dandelion': 'Yellow flower', 'Rose': 'Rose', 'Sunflower': 'Sunflower', 'Tulip': 'Tulip'}

# ====================== LOAD MODEL ======================
@st.cache_resource
def load_model():
    extractor = tf.keras.models.load_model(PATH_EXTRACTOR)
    with open(PATH_SCALER, 'rb') as f:
        scaler = pickle.load(f)
    classifier = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(768,)),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    classifier.load_weights(PATH_WEIGHTS)
    return extractor, scaler, classifier

extractor, scaler, classifier = load_model()

# ====================== PREDICTION ======================
def predict(img):
    img = img.resize(IMG_SIZE)
    x = np.expand_dims(img, axis=0)
    feats = extractor.predict(x, verbose=0)
    feats = scaler.transform(feats)
    probs = classifier.predict(feats, verbose=0)[0]
    idx = np.argmax(probs)
    return CLASSES[idx], probs[idx], probs

# ====================== UI ======================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1822/1822167.png", width=100)
    st.title("Flower AI Assistant")
    st.markdown("---")
    st.info("**Model:** Vision Transformer (ViT)\n\n**5 loài hoa:**")
    for c in CLASSES:
        st.write(f"{EMOJIS[c]} {c}")

st.title("Flower Recognition AI")
st.markdown("##### Tải lên một bông hoa – AI sẽ nói tên ngay lập tức!")

file = st.file_uploader("Chọn ảnh hoa", type=["jpg", "jpeg", "png", "webp"])

if file and extractor:
    img = Image.open(file).convert("RGB")
    c1, c2 = st.columns([1, 1.2])
    
    with c1:
        st.image(img, use_column_width=True)
    
    if st.button("Analyze Now", type="primary", use_container_width=True):
        with st.spinner("Đang nhận diện..."):
            label, conf, probs = predict(img)
        
        with c2:
            st.success("Hoàn tất!")
            st.markdown(f"<h2 style='text-align:center'>{EMOJIS[label]} {label}</h2>", unsafe_allow_html=True)
            st.metric("Độ tin cậy", f"{conf:.1%}")
            st.progress(conf)
            
            df = pd.DataFrame({"Hoa": CLASSES, "Xác suất": probs})
            st.bar_chart(df.set_index("Hoa"), color="#FF69B4")

else:
    st.info("Đang tải model hoặc chờ bạn upload ảnh...")
