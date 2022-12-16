import math
import numpy as np
import time
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from pynput.keyboard import Key, Controller

keyboard = Controller()

cam = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.2)
mpDraw = mp.solutions.drawing_utils
hdX = []
hdY = []
for i in range(1000):
    _, img = cam.read()
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_RGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 0:
                    hdX.append(cx)
                    hdY.append(cy)
                    break
    
plt.plot(hdX)
plt.show(block=True)
    

    
