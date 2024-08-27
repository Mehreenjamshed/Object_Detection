# Install required packages
!pip install ultralytics opencv-python-headless matplotlib

# Import necessary libraries
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Provide the path to the uploaded video
video_path = '/content/video.mp4'  # Adjust the path if necessary

# Open the video for processing
cap = cv2.VideoCapture(video_path)

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform object detection on the frame
    results = model(frame)

    # Draw bounding boxes on the frame
    annotated_frame = results[0].plot()
    
    # Display the frame
    plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

# Release the video capture object
cap.release()

print("Finished processing video.")
