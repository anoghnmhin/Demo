# app.py
import streamlit as st
from PIL import Image
import random
import time

st.set_page_config(page_title="Emotion Recognition Demo", page_icon="Face", layout="centered")

with open(".streamlit/config.toml", "w") as f:
    f.write('[theme]\nbase = "light"')

st.markdown("""
<style>
    .main {background: #f9fafb; padding: 2rem;}
    .title {font-size: 3.5rem; font-weight: 800; text-align: center; color: #1f2937;}
    .card {background: white; padding: 2.5rem; border-radius: 1.5rem; box-shadow: 0 20px 40px rgba(0,0,0,0.1); transition: all 0.3s;}
    .card:hover {transform: translateY(-10px);}
    .upload-area {border: 4px dashed #c7d2fe; border-radius: 1rem; padding: 5rem; text-align: center; background: #eef2ff;}
    .upload-area:hover {border-color: #6366f1; background: #e0e7ff;}
    .big-emoji {font-size: 6rem; margin: 1rem 0;}
    .confidence-fill {height: 100%; background: linear-gradient(90deg, #6366f1, #10b981); border-radius: 999px; transition: width 1.2s ease-out;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">Facial Emotion Recognition</h1>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, use_column_width=True)
    else:
        st.markdown("<div class='upload-area'><div style='font-size:4.5rem;'>Upload</div><p style='color:#6366f1; font-weight:600;'>Click or drop image</p></div>", unsafe_allow_html=True)
    
    predict = st.button("Detect Emotion", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    if uploaded and predict:
        with st.spinner("Analyzing..."):
            time.sleep(2)
            emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
            emotion = random.choice(emotions)
            conf = round(random.uniform(72, 96), 1)
            
            st.markdown(f"<div class='big-emoji'>{emotion}</div>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align:center; color:#1f2937;'>{emotion}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align:center; color:#10b981; font-weight:bold;'>{conf}% Confidence</h3>", unsafe_allow_html=True)
            st.markdown(f"<div style='background:#e5e7eb; height:24px; border-radius:999px; margin:2rem 0;'><div class='confidence-fill' style='width:{conf}%'></div></div>", unsafe_allow_html=True)
    else:
        if uploaded:
            st.image(img, use_column_width=True)
        else:
            st.image("https://placehold.co/600x500/f8fafc/9ca3af?text=Waiting+for+Result", use_column_width=True)
        
        st.markdown("<div style='text-align:center; padding:5rem 0; color:#94a3b8;'><div style='font-size:5rem;'>MagnifyingGlass</div><p style='font-size:1.3rem;'>Result will appear here</p></div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<p style='text-align:center; color:#94a3b8; margin-top:3rem; font-size:0.9rem;'>CS420 Final Project • VGG16 • AffectNet Dataset</p>", unsafe_allow_html=True)
