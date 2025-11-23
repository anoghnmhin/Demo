# app.py
import streamlit as st
from PIL import Image
import numpy as np
import random
import time

# ============================
# Cấu hình trang
# ============================
st.set_page_config(
    page_title="Emotion Recognition Demo",
    page_icon="Face",
    layout="centered"
)

# ============================
# CSS đẹp y hệt HTML (Tailwind style)
# ============================
st.markdown("""
<style>
    .main {background: #f9fafb; padding: 2rem;}
    .title {font-size: 3.5rem; font-weight: 800; text-align: center; color: #1f2937;}
    .subtitle {text-align: center; color: #6b7280; margin-bottom: 3rem;}
    .card {
        background: white;
        padding: 2rem;
        border-radius: 1.5rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }
    .card:hover {transform: translateY(-8px);}
    .upload-box {
        border: 4px dashed #d1d5db;
        border-radius: 1rem;
        padding: 4rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s;
    }
    .upload-box:hover {border-color: #6366f1;}
    .big-emoji {font-size: 5.5rem; margin: 1rem 0;}
    .confidence-bar {
        height: 20px;
        background: #e5e7eb;
        border-radius: 999px;
        overflow: hidden;
        margin: 2rem 0;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #6366f1, #10b981);
        border-radius: 999px;
        width: 0%;
        transition: width 1.2s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# Tiêu đề
# ============================
st.markdown('<h1 class="title">Facial Emotion Recognition</h1>', unsafe_allow_html=True)

# ============================
# Layout 2 cột
# ============================
col1, col2 = st.columns(2)

# Cột trái: Upload + Nút
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    else:
        st.markdown("""
        <div class='upload-box'>
            <div style='font-size: 4rem;'>Upload</div>
            <p style='color: #9ca3af; margin-top: 1rem;'>Click or drop an image</p>
        </div>
        """, unsafe_allow_html=True)
    
    predict_btn = st.button("Detect Emotion", type="primary", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Cột phải: Kết quả
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    if uploaded_file and predict_btn:
        with st.spinner("Analyzing emotion..."):
            time.sleep(1.8)  # Giả lập thời gian xử lý
            
            emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
            emoji_map = {
                "Angry": "Angry", "Disgust": "Disgusted", "Fear": "Fearful",
                "Happy": "Happy", "Sad": "Sad", "Surprise": "Surprised", "Neutral": "Neutral"
            }
            emotion = random.choice(emotions)
            confidence = round(random.uniform(70, 95), 1)
            
            st.markdown(f"<div class='big-emoji'>{emoji_map[emotion]}</div>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align:center; color:#1f2937;'>{emotion}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center; font-size:1.5rem; color:#10b981; font-weight:bold;'>{confidence}% confidence</p>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class='confidence-bar'>
                <div class='confidence-fill' style='width: {confidence}%'></div>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        if uploaded_file:
            st.image(image, use_column_width=True)
        else:
            st.image("https://placehold.co/500x400/f1f5f9/64748b?text=No+Image", use_column_width=True)
            
        st.markdown("""
        <div style='text-align:center; padding: 4rem 0; color: #9ca3af;'>
            <div style='font-size: 4rem; margin-bottom: 1rem;'>MagnifyingGlass</div>
            <p style='font-size: 1.2rem;'>Result will appear here</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ============================
# Footer
# ============================
st.markdown("<p style='text-align:center; color:#6b7280; margin-top: 3rem; font-size: 0.9rem;'>"
            "CS420 Final Project • VGG16 • AffectNet Dataset</p>", unsafe_allow_html=True)
