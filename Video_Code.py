import torch
import cv2
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load YOLOv5 pretrained model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Quick hack for demo: rename 'bird' to 'Fish'
FISH_CLASSES = ['Fish']

# Open webcam
cap = cv2.VideoCapture(0)  # 0 = default camera

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run detection
    results = model(frame)

    # Convert results to pandas dataframe
    df = results.pandas().xyxy[0]

    # Rename 'bird' to 'Fish'
    df['name'] = df['name'].replace('bird', 'Fish')

    # Check if fish detected
    fish_detected = any(name == 'Fish' for name in df['name'])

    # Voice alert
    if fish_detected:
        print("Fish Detected!")
        engine.say("Fish")
    else:
        print("No Fish Detected")
        engine.say("No Fish")
    engine.runAndWait()

    # Draw bounding boxes
    for i in range(len(df)):
        x1, y1, x2, y2 = int(df['xmin'][i]), int(df['ymin'][i]), int(df['xmax'][i]), int(df['ymax'][i])
        label = df['name'][i]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display video
    cv2.imshow("Live Fish Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
