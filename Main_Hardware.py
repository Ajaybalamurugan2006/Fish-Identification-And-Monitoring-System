import tkinter as tk
from tkinter import filedialog
import torch
import cv2
from PIL import Image, ImageTk
import pyttsx3
import threading
import serial
import time  # <--- added for Arduino initialization

# ===================== Arduino Setup =====================
arduino_connected = False
try:
    ser = serial.Serial('COM3', 9600, timeout=1)  # change COM port
    time.sleep(2)  # wait for Arduino
    arduino_connected = True
except Exception as e:
    print("Arduino not connected:", e)
    ser = None

# Initialize TTS engine
engine = pyttsx3.init()

# Load YOLOv5 pretrained model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Quick hack for demo: rename 'bird' to 'Fish'
FISH_CLASSES = ['Fish']

# GUI setup
root = tk.Tk()
root.title("Fish Identification System")
root.geometry("1000x750")
root.configure(bg="#1E1E2F")  # dark background

# Result Label
result_label = tk.Label(root, text="Result: ", font=("Arial", 22, "bold"), bg="#1E1E2F", fg="#FFD700")
result_label.pack(pady=15)

# Canvas to display image/video
canvas = tk.Label(root, bg="#1E1E2F")
canvas.pack(pady=10)

# Event to stop video safely
stop_event = threading.Event()

# Function to detect fish in a frame
def detect_fish(frame):
    results = model(frame)
    df = results.pandas().xyxy[0]
    df['name'] = df['name'].replace('bird', 'Fish')  # rename for demo
    fish_detected = any(name == 'Fish' for name in df['name'])
    # Draw bounding boxes
    for i in range(len(df)):
        x1, y1, x2, y2 = int(df['xmin'][i]), int(df['ymin'][i]), int(df['xmax'][i]), int(df['ymax'][i])
        label = df['name'][i]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    return frame, fish_detected

# Image Mode
def select_image():
    stop_event.set()  # stop video if running
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return
    frame = cv2.imread(file_path)
    frame, fish_detected = detect_fish(frame)
    if fish_detected:
        result_label.config(text="Result: Fish Identified ", fg="#00FF00")
        engine.say("Fish")
        if arduino_connected: ser.write(b'1')  # <--- send 1
    else:
        result_label.config(text="Result: No Fish Identified", fg="#FF4500")
        engine.say("No Fish")
        if arduino_connected: ser.write(b'2')  # <--- send 2
    engine.runAndWait()
    # Convert to PIL Image and show on canvas
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    canvas.imgtk = imgtk
    canvas.config(image=imgtk)

# Video Mode (Live Webcam)
def start_video():
    stop_event.clear()
    cap = cv2.VideoCapture(0)

    def video_loop():
        prev_fish_state = None
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            frame, fish_detected = detect_fish(frame)
            # Update result only on state change
            if fish_detected != prev_fish_state:
                if fish_detected:
                    result_label.config(text="Result: Fish Identified", fg="#00FF00")
                    engine.say("Fish")
                    if arduino_connected: ser.write(b'1')  # <--- send 1
                else:
                    result_label.config(text="Result: No Fish Identified", fg="#FF4500")
                    engine.say("No Fish")
                    if arduino_connected: ser.write(b'2')  # <--- send 2
                engine.runAndWait()
                prev_fish_state = fish_detected
            # Convert to PIL Image and display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            canvas.imgtk = imgtk
            canvas.config(image=imgtk)
        cap.release()

    threading.Thread(target=video_loop, daemon=True).start()

# Stop Video Button
def stop_video():
    stop_event.set()
    result_label.config(text="Video Stopped", fg="#FFD700")

# Buttons Frame
btn_frame = tk.Frame(root, bg="#1E1E2F")
btn_frame.pack(pady=20)

btn_style = {"width": 20, "height": 2, "font": ("Arial", 14, "bold"), "bg": "#4B0082", "fg": "white", "activebackground": "#6A0DAD"}

image_btn = tk.Button(btn_frame, text="Select Image", command=select_image, **btn_style)
image_btn.grid(row=0, column=0, padx=15)

video_btn = tk.Button(btn_frame, text="Start Live Video", command=start_video, **btn_style)
video_btn.grid(row=0, column=1, padx=15)

stop_btn = tk.Button(btn_frame, text="Stop Video", command=stop_video, **btn_style)
stop_btn.grid(row=0, column=2, padx=15)

root.mainloop()
