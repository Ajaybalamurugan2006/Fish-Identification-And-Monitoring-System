import torch
import cv2
import pyttsx3

# Initialize TTS engine
engine = pyttsx3.init()

# Load YOLOv5 model (pretrained)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Webcam input
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Get results as pandas dataframe
    df = results.pandas().xyxy[0]

    # Quick hack: rename 'bird' detections to 'Fish'
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

    # Show the frame
    cv2.imshow("Fish Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
