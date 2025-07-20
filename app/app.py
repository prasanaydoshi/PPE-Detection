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
    # Get the file extension from uploaded file
    file_extension = uploaded.name.split('.')[-1]
    
    # Create temp file WITH extension
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}')
    tmp.write(uploaded.read())
    tmp.close()  # Close before YOLO reads it
    source_path = tmp.name
    
    # Inference
    results = model.predict(source_path, conf=conf, save=False)
    for r in results:
        annotated = r.plot()
        if len(r.orig_shape) == 3:  # image (height, width, channels)
            st.image(annotated[..., ::-1], channels="RGB")
        else:  # video - this logic might need adjustment
            st.video(cv2.imencode(".mp4", annotated)[1].tobytes())
