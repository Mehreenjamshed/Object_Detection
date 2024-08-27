import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Streamlit app title and description
st.title("YOLOv8 Object Detection on Video")
st.write("Upload a video and the model will perform object detection frame by frame.")

# Video upload in Streamlit
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# If a video is uploaded
if uploaded_video is not None:
    # Save the uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    # Open the video using OpenCV
    cap = cv2.VideoCapture(tfile.name)

    # Process the video frame by frame
    stframe = st.empty()  # Placeholder for displaying frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection on the frame
        results = model(frame)

        # Draw bounding boxes on the frame
        annotated_frame = results[0].plot()

        # Display the frame in Streamlit
        stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")

    # Release the video capture object
    cap.release()
    st.success("Finished processing the video.")
