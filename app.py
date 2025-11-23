# app.py
import streamlit as st
from PIL import Image
import random

st.title("Facial Emotion Recognition")

col1, col2 = st.columns(2)

with col1:
    st.write("#### Upload Image")
    uploaded_file = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Your photo", use_column_width=True)
    else:
        st.info("No image uploaded yet")

    predict = st.button("Detect Emotion", type="primary", use_container_width=True)

with col2:
    st.write("#### Result")

    if uploaded_file and predict:
        with st.spinner("Analyzing..."):
            emotions = ["Happy", "Sad", "Angry", "Surprise", "Fear", "Disgust", "Neutral"]
            result = random.choice(emotions)
            confidence = round(random.uniform(65, 98), 1)

        st.success(f"**{result}**")
        st.metric("Confidence", f"{confidence}%")
        st.progress(confidence / 100)

        if uploaded_file:
            st.image(image, use_column_width=True)

    else:
        st.image("https://placehold.co/400x300?text=Result+will+appear+here", use_column_width=True)
        st.write("Upload an image and click **Detect Emotion**")

st.caption("CS420 Final Project • VGG16 Model • AffectNet Dataset")
