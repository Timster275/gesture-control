

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyautogui as pyautogui
from time import sleep


mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

historyPredictions = []
WAITING_TIME = 1

model = load_model('gesture_detection_model.hdf5')

classNames = {
    0: "Stop",
    1: "Fist up",
    2: "OK",
    3: "Down",
    4: "Fist",
    5: "Left",
    6: "Right"
    
}


def do_action():
    prediction = max(set(historyPredictions), key=historyPredictions.count)
    print(prediction, classNames[prediction])
    action_done = False
    if(prediction == 0):
        do_action_open_hand()
        action_done = True

    if(action_done):
        sleep(WAITING_TIME)


def get_active_window():
    return pyautogui.getActiveWindow().title


def do_action_open_hand():
    if get_active_window().find("PowerPoint") != -1:
        pyautogui.press('right')
    elif get_active_window().find("Edge") != -1 or get_active_window().find("Chrome") != -1:
        pyautogui.keyDown('ctrl')
        pyautogui.press('tab')
        pyautogui.keyUp('ctrl')
    else:
        pyautogui.press('playpause')


def predict_class(landmarks):
    landmarks = np.array(landmarks)

    landmarks = landmarks/abs(max(landmarks, key=abs))

    landmarks = landmarks.reshape(42, 1)
    prediction = model.predict(np.array([landmarks]))
    classID = np.argmax(prediction)
    className = classNames[classID]
    historyPredictions.append(classID)

    return className, classID


def create_landmarks(multi_hand_landmarks):
    landmarks = []
    for handslms in multi_hand_landmarks:
        for lm in handslms.landmark:
            landmarks.append(lm.x)
            landmarks.append(lm.y)

    return landmarks

def draw_landmarks(framergb, handslms):
    for handslms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(framergb, handslms, mpHands.HAND_CONNECTIONS)


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
        landmarks = create_landmarks(result.multi_hand_landmarks)

        draw_landmarks(framergb, result.multi_hand_landmarks)

        className, classId = predict_class(landmarks)

        if(len(historyPredictions) > 9):
            # do_action()
            historyPredictions.clear()
    cv2.putText(framergb, className, (450, 220), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Output", framergb)

    if cv2.waitKey(1) == ord('q'):
        break
    # click on exit button to exit
    # if cv2.getWindowProperty('Output', cv2.WND_PROP_VISIBLE) < 1:
    #     break

cap.release()

cv2.destroyAllWindows()
