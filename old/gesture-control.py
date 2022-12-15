# Was wurde verwendet:
# OpenCV => Kamera aktivieren
# math => um von deltax und deltay radiant zu bekommen
# mediapipe => Hand detection
# pynput => Keyboard shortcuts ausführen wie (alt + tab, Windows + D)
# models => Datei mit Zahlen für die Richtung der Hand
# time => um jede halbe Sekunde eine Position auszulesen (delay)

import math
import cv2
import time
import mediapipe as mp
import models
from pynput.keyboard import Key, Controller


keyboard = Controller()

cam = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1,
                      min_detection_confidence=0.5, min_tracking_confidence=0.2)
mpDraw = mp.solutions.drawing_utils

pTime = 0


def getHandPosition(results):
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 0:
                    position.append((cx, cy))
                if id == 12:
                    position2.append((cx, cy))
            #mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
        return True
    else:
        return False


position = []
position2 = []


def movement_direction(x_delta, y_delta, threshhold=10):
    if abs(x_delta) > threshhold or abs(y_delta) > threshhold:
        degree = math.atan2(y_delta, x_delta)

        if -0.875 * math.pi <= degree < -0.625 * math.pi:
            direction = models.UP_RIGHT
        elif -0.625 * math.pi <= degree < -0.375 * math.pi:
            direction = models.UP
        elif -0.375 * math.pi <= degree < -0.125 * math.pi:
            direction = models.UP_LEFT
        elif -0.125 * math.pi <= degree < 0.125 * math.pi:
            direction = models.LEFT
        elif 0.125 * math.pi <= degree < 0.375 * math.pi:
            direction = models.DOWN_LEFT
        elif 0.375 * math.pi <= degree < 0.625 * math.pi:
            direction = models.DOWN
        elif 0.625 * math.pi <= degree < 0.875 * math.pi:
            direction = models.DOWN_RIGHT
        else:
            direction = models.RIGHT

        return direction
    else:
        return None


isPressed = False

while True:
    _, img = cam.read()
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_RGB)
    exists = getHandPosition(results)
    if len(position) >= 2:
        x0 = position[0][0]
        x1 = position[1][0]
        y0 = position[0][1]
        y1 = position[1][1]
        y2 = position2[0][1]
        y3 = position2[1][1]
        position.clear()
        position2.clear()
        deltax = x1 - x0
        deltay = y1 - y0
        pRes2 = movement_direction(deltax, deltay)
        pRes3 = movement_direction(deltax, deltay, 30)
        # print(y1- y3)
        # print("#####")
        # print(y1 - y2)
        # print()
        if (y1 - y3) < 50 and (y1 - y2) < 50:
            keyboard.press(Key.cmd_l)
            keyboard.press('d')
            keyboard.release(Key.cmd_l)
            keyboard.release('d')
        elif (y1 - y2) > 200:
            if isPressed == False:
                keyboard.press(Key.alt)
                keyboard.press(Key.tab)
                keyboard.release(Key.tab)
                isPressed = True

        if pRes2 == 3 and isPressed:
            keyboard.press(Key.right)
            keyboard.release(Key.right)
        if pRes2 == 2 and isPressed:
            keyboard.press(Key.left)
            keyboard.release(Key.left)
    if exists == False and isPressed:
        keyboard.release(Key.alt)
        keyboard.release(Key.cmd_l)
        isPressed = False

    time.sleep(0.3)

    # Ausgabe Kamera mit frames wird gerade nicht verwendet
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Test", img)
    cv2.waitKey(1)
