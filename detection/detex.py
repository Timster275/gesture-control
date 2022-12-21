


import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

import random

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
# model = load_model('mp_hand_gesture')
model = load_model('model.hdf5')
# Load class names
# f = open('gesture.names', 'r')
classNames = {
    0: "Flat Hand",
    1: "O-Kay",
    2: "Hand Side",
    3: "Three Fingers"
}

# Initialize the webcam
cap = cv2.VideoCapture(0)
i = 0
while True:
    # Read each frame from the webcam
    _, frame = cap.read()


    image_width, image_height = frame.shape[1], frame.shape[0]

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)

    framergb = frame
    result = hands.process(framergb)

    
    className = ''

    # post process the result
    if result.multi_hand_landmarks:

        landmarks = []
        for handslms in result.multi_hand_landmarks:
            i = 0
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * image_width)
                lmy = int(lm.y * image_height)
                if i == 0:
                    temp_x = lmx
                    temp_y = lmy
                    lmx = 0
                    lmy = 0
                else:
                    lmx = lmx - temp_x
                    lmy = lmy - temp_y

                # landmarks.append([lmx, lmy])
                landmarks.append(lm.x)
                landmarks.append(lm.y)

                i = i + 1

            
            mpDraw.draw_landmarks(framergb, handslms, mpHands.HAND_CONNECTIONS)

            landmarks = np.array(landmarks)

            landmarks = landmarks/abs(max(landmarks, key=abs))

            landmarks = landmarks.reshape(42, 1)
            prediction = model.predict(np.array([landmarks]))


            classID = np.argmax(prediction)
            className = classNames[classID]
            
            print(classID)
            print(className)
            # emoji_path = "emojis/" + str(classID) + ".png"

            # overlay = cv2.imread(emoji_path)
            # if classID == 0:
            #     overlay = cv2.resize(overlay, (150, 150))
            # else:
            #     overlay = cv2.resize(overlay, (100, 100))
            # h, w = overlay.shape[:2]
            # shapes = np.zeros_like(framergb, np.uint8)
            # shapes[0:h, 0:w] = overlay
            # alpha = 0.8
            # mask = shapes.astype(bool)
            # framergb[mask] = cv2.addWeighted(shapes, alpha, shapes, 1 - alpha, 0)[mask]
    # Show the final output
    cv2.putText(framergb, className, (450, 220), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Output", framergb)

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()