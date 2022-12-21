


import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

import random

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils


model = load_model('model.hdf5')

classNames = {
    0: "Flat Hand",
    1: "O-Kay",
    2: "Hand Side",
    3: "Three Fingers",
    4: "Fist"
}


cap = cv2.VideoCapture(0)
i = 0
while True:
    _, frame = cap.read()


    image_width, image_height = frame.shape[1], frame.shape[0]


    frame = cv2.flip(frame, 1)

    framergb = frame
    result = hands.process(framergb)

    
    className = ''


    if result.multi_hand_landmarks:

        landmarks = []
        for handslms in result.multi_hand_landmarks:
            i = 0
            for lm in handslms.landmark:
                # ignore
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
                # dont ignore 
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
            
    cv2.putText(framergb, className, (450, 220), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Output", framergb)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()