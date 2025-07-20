# app/app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2, tempfile, numpy as np

@st.cache_resource  # model loads once per session
def load_model(local_weight: str = "models/best.pt"):
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
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(uploaded.read())                # save to disk for YOLO
    source_path = tmp.name

    # Inference ➜ `predict()` auto‑detects img vs. video
    results = model.predict(source_path, conf=conf, save=False)
    for r in results:                         # iterate frames if video
        # Draw bounding boxes
        annotated = r.plot()                  # returns BGR np.array
        if r.orig_shape[2] == 3:              # image
            st.image(annotated[..., ::-1], channels="RGB")
        else:                                 # video frame
            st.video(cv2.imencode(".mp4", annotated)[1].tobytes())
