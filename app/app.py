# app/app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2, tempfile, numpy as np

@st.cache_resource  # model loads once per session
def load_model(local_weight: str = "model/best.pt"):
    return YOLO(local_weight)                 # ➜ PyTorch backend

model = load_model()
st.title("🦺 PPE‑Detector")
st.write(
    "Upload an **image** or **video** (≤200 MB) and get bounding‑box predictions powered by YOLOv5."
)

conf = st.slider("Confidence threshold", 0.05, 1.0, 0.25, 0.05)
uploaded = st.file_uploader(
    "Choose image / video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"]
)

if uploaded:
    file_extension = uploaded.name.split('.')[-1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}')
    tmp.write(uploaded.read())
    tmp.close()
    source_path = tmp.name
    
    try:
        results = model.predict(source_path, conf=conf, save=False)
        
        # Check if it's an image or video based on file extension
        if file_extension in ['jpg', 'jpeg', 'png']:
            # Handle images
            for r in results:
                annotated = r.plot()
                st.image(annotated[..., ::-1], channels="RGB")
        else:
            # Handle videos - for now, just show first frame
            # (proper video handling is more complex)
            for r in results:
                annotated = r.plot()
                st.image(annotated[..., ::-1], channels="RGB")
                break  # Just show first frame for now
                
    except Exception as e:
        st.error(f"Error processing file: {e}")
    finally:
        # Clean up
        import os
        if os.path.exists(source_path):
            os.unlink(source_path)
