import cv2
import tkinter as tk
from tkinter import ttk
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from PIL import Image, ImageTk
import ttkthemes as th
from gtts import gTTS
from playsound import playsound

# Function to handle the "Clear" button click event
def next_predictions():
    global final_prediction
    global cleared_predictions
    if final_prediction is not None:
        cleared_predictions.append(final_prediction)
        final_prediction_label.config(text=f"Prediction cleared: {final_prediction}")
    final_prediction = None

# Function to handle the "Exit" button click event
def exit_program():
    cap.release()
    cv2.destroyAllWindows()
    root.quit()

# Function to speak the final prediction
def speak_prediction():
    global final_prediction
    separator = " "  # This is the character that will separate the elements in the string

    # Use the join() method to concatenate the elements with the separator
    result_string = separator.join(final_prediction)
    if final_prediction:
        tts = gTTS(text=result_string, lang='en')
        tts.save("C:\\Users\\Acer\\PycharmProjects\\ChemistryLightsolver\\prediction.mp3")
        playsound("prediction.mp3")

# Function to update the UI with the webcam feed
def update():
    global imgResize
    global final_prediction
    global prediction_display

    success, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    img = cv2.flip(img, 1)  # Flip horizontally for mirror effect

    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            siz = (wCal, imgSize)
            try:
                imgResize = cv2.resize(imgCrop, siz, interpolation=cv2.INTER_AREA)
            except:
                print("hand not in frame")
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            final_prediction_label.config(text=f"Final Prediction: {labels[index]}")

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            try:
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            except:
                print("hand not in frame")
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            final_prediction_label.config(text=f"Final Prediction: {labels[index]}")


        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset), (23, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 1.7, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (23, 255, 0), 4)

        # Store the prediction as the final prediction
        final_prediction = labels[index]

    key = cv2.waitKey(1)

    if key == ord('c'):  # Press 'c' to clear the previous prediction
        if final_prediction is not None:
            cleared_predictions.append(final_prediction)
            print("Prediction cleared:", final_prediction)
        final_prediction = None

    if final_prediction is not None:
        prediction_display = f"Final Prediction: "
        if cleared_predictions:
            prediction_display += f"  {''.join(cleared_predictions)}"
        final_prediction_label.config(text=f" {prediction_display}")

    # Convert the updated image to a PhotoImage object for Tkinter
    img_pil = Image.fromarray(imgOutput)
    img_tk = ImageTk.PhotoImage(image=img_pil)

    # Update the UI with the updated webcam feed
    img_label.config(image=img_tk)
    img_label.img = img_tk
    img_label.after(10, update)  # Call update function every 10 milliseconds

# Create the main GUI window
root = th.ThemedTk()
root.get_themes()
root.set_theme("arc")  # You can change the theme here

root.title("Hand Gesture Classifier")

# Create a frame for the video display
frame = ttk.Frame(root)
frame.pack(pady=10)

# Create a custom style for labels and buttons with the desired font and size
custom_style = ttk.Style()
custom_style.configure("Custom.TLabel", font=("Times New Roman", 24))
custom_style.configure("Custom.TButton", font=("Times New Roman", 24))

# Create a label for displaying the final prediction with custom style
final_prediction_label = ttk.Label(root, text=" ", style="Custom.TLabel")
final_prediction_label.pack(pady=5)

# Create "Clear" and "Exit" buttons with updated styles
next_button = ttk.Button(root, text="Next", command=next_predictions, style="Custom.TButton")
next_button.pack(side=tk.LEFT, padx=10)
exit_button = ttk.Button(root, text="Exit", command=exit_program, style="Custom.TButton")
exit_button.pack(side=tk.RIGHT, padx=10)

# Create a "Speak" button with the custom style
speak_button = ttk.Button(root, text="Speak", command=speak_prediction, style="Custom.TButton")
speak_button.pack(side=tk.LEFT, padx=10)

# Create a label for displaying the webcam feed
img_label = ttk.Label(root)
img_label.pack()

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 15
imgSize = 300

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", " ",
          "O", "P", "Q", "R", "S", "T", "    ", "U", "V", "W", "X", "Y", "Z"]

arr = []
cleared_predictions = []  # To store cleared predictions
final_prediction = None  # To store the final prediction

# Start the webcam feed update
update()

# Start the GUI main loop
root.mainloop()