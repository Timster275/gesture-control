import math
import numpy as np
import time
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import json
import sys

cam = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.2)
mpDraw = mp.solutions.drawing_utils
hdX1 = []
hdY1 = []
hdZ1 = []

# get the gesture from the arguments
gesture = sys.argv[1]
for i in range(200):
    _, img = cam.read()
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_RGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            hmX = []
            # hmY = []
            # hmZ = []
            hmX.append(sys.argv[2])
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy, cz  = lm.x, lm.y, lm.z

                hmX.append(cx)
                hmX.append(cy)
                hmX.append(cz)

            hdX1.append(hmX)

class DF:
    def __init__(self, name, x):#, y, z):
        self.name = name
        self.x = x
        # self.y = y
        # self.z = z
    def save(self):
        with open("gesture_data_new/" + self.name + ".dat", "w+") as f:
            f.write(json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4))

df = DF(gesture, hdX1)#, hdY1, hdZ1)
df.save()
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(hdX, hdY, hdZ)
# plt.scatter(hdX1, hdY1, label="Hands")
# plt.show(block=True)

    
