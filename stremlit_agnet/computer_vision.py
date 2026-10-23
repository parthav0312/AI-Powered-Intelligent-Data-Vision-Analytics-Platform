import streamlit as st
import cv2
import numpy as np
import torch
import time
from PIL import Image
import easyocr
import mlflow
import tempfile
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------
st.set_page_config(page_title="Computer Vision Pro Suite", page_icon="📸", layout="wide")
st.title("📸 Computer Vision  OpenCV + PyTorch + MLflow")
st.caption("Advanced Image Processing • Object Detection • OCR • Experiment Tracking • Live Camera")

# ----------------------------------------------------------
# Sidebar Controls
# ----------------------------------------------------------
with st.sidebar:
    st.header("🔧 Configuration")
    mode = st.radio(
        "Select Task",
        [
            "Image Enhancement",
            "Object Detection (YOLOv5)",
            "OCR Text Extraction",
            "Face Detection",
            "Live Camera Detection",
        ],
        index=0,
    )

    mlflow_logging = st.checkbox("Enable MLflow Tracking", value=False)

# ----------------------------------------------------------
# MLflow Setup
# ----------------------------------------------------------
if mlflow_logging:
    mlflow.set_experiment("Computer_Vision")
    run = mlflow.start_run(run_name=mode)
    start_time = time.time()

# ----------------------------------------------------------
# File Upload (not needed for Live mode)
# ----------------------------------------------------------
uploaded_file = None
if mode != "Live Camera Detection":
    uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("Please upload an image to begin.")
        st.stop()

    image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    col1, col2 = st.columns(2)
    col1.image(image_rgb, caption="Original Image", use_container_width=True)

# ----------------------------------------------------------
# Mode 1 – Image Enhancement
# ----------------------------------------------------------
if mode == "Image Enhancement":
    st.subheader("🧠 Image Enhancement")
    blur = st.slider("Gaussian Blur", 0, 15, 3, step=2)
    brightness = st.slider("Brightness Adjustment", -50, 50, 0)
    contrast = st.slider("Contrast Adjustment", 1.0, 3.0, 1.0)

    enhanced = cv2.GaussianBlur(image_rgb, (blur, blur), 0)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=contrast, beta=brightness)

    col2.image(enhanced, caption="Enhanced Image", use_container_width=True)

    if mlflow_logging:
        mlflow.log_param("blur", blur)
        mlflow.log_param("contrast", contrast)
        mlflow.log_param("brightness", brightness)

# ----------------------------------------------------------
# Mode 2 – Object Detection (YOLOv5)
# ----------------------------------------------------------
elif mode == "Object Detection (YOLOv5)":
    st.subheader("🎯 Object Detection using YOLOv5")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True).to(device)
    results = model(image_rgb)
    detected = np.squeeze(results.render())
    col2.image(detected, caption="Detected Objects", use_container_width=True)
    st.dataframe(results.pandas().xyxy[0])

    if mlflow_logging:
        mlflow.log_param("model", "yolov5s")
        mlflow.log_metric("detections", len(results.pandas().xyxy[0]))

# ----------------------------------------------------------
# Mode 3 – OCR Text Extraction
# ----------------------------------------------------------
elif mode == "OCR Text Extraction":
    st.subheader("📄 Optical Character Recognition (OCR)")
    reader = easyocr.Reader(["en"])
    result = reader.readtext(image_rgb)
    ocr_img = image_rgb.copy()

    for (bbox, text, prob) in result:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        cv2.rectangle(ocr_img, tuple(map(int, top_left)), tuple(map(int, bottom_right)), (0, 255, 0), 2)
        cv2.putText(ocr_img, text, (int(top_left[0]), int(top_left[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    col2.image(ocr_img, caption="OCR Output", use_contanier_width=True)
    extracted_text = [text for (_, text, _) in result]
    st.write("**Extracted Text:**", extracted_text)

    if mlflow_logging:
        mlflow.log_param("ocr_texts", len(extracted_text))
        mlflow.log_text("\n".join(extracted_text), "ocr_output.txt")

# ----------------------------------------------------------
# Mode 4 – Face Detection
# ----------------------------------------------------------
elif mode == "Face Detection":
    st.subheader("😎 Face Detection (Haar Cascade)")
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    face_img = image_rgb.copy()

    for (x, y, w, h) in faces:
        cv2.rectangle(face_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    col2.image(face_img, caption=f"Detected {len(faces)} Face(s)", use_container_width=True)

    if mlflow_logging:
        mlflow.log_metric("faces_detected", len(faces))

# ----------------------------------------------------------
# Mode 5 – Live Camera Detection
# ----------------------------------------------------------
elif mode == "Live Camera Detection":
    st.subheader("🎥 Real-time Object Detection (YOLOv5 + OpenCV)")

    # ✅ Unique key for start button
    run_live = st.button("Start Camera", key="start_camera_btn")

    if run_live:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True).to(device)
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        prev_time = time.time()

        if mlflow_logging:
            mlflow.log_param("model", "yolov5s_realtime")

        # ✅ Use session_state to persist camera running state
        st.session_state["camera_running"] = True

        # ✅ Sidebar stop button with unique key
        stop_stream = st.sidebar.button("Stop Camera", key="stop_camera_btn")

        while cap.isOpened() and st.session_state.get("camera_running", True):
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Unable to access camera.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame)
            annotated = np.squeeze(results.render())

            fps = 1.0 / (time.time() - prev_time)
            prev_time = time.time()

            # ✅ Modern Streamlit param
            stframe.image(annotated, channels="RGB", use_container_width=True)
            st.sidebar.info(f"⚡ FPS: {fps:.2f}")

            # ✅ Check for stop flag each iteration
            if stop_stream:
                st.session_state["camera_running"] = False
                break

        cap.release()
        st.success("✅ Camera stopped.")


# ----------------------------------------------------------
# MLflow End Run
# ----------------------------------------------------------
if mlflow_logging:
    mlflow.log_metric("runtime_sec", round(time.time() - start_time, 2))
    mlflow.end_run()
