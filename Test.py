import torch
import cv2

# Load model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Read image
img = cv2.imread('test5.jpg')  # replace with your image path

# Inference
results = model(img)

# Show results
results.show()  # opens window with detections
results.print()  # prints detected objects
