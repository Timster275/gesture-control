# import necessary packages
from gesture_methods import do_action, move_cursor, set_mouse_up
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from common_methods import load_class_names
from gesture_methods import settings
import multiprocessing

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')


# Load class names
classNames = load_class_names('gesture.names')

historyPredictions = []


def get_landmarks(hands_lms):
    landmarks = []
    for lm in hands_lms.landmark:
        # print(id, lm)
        lmx = int(lm.x * x)
        lmy = int(lm.y * y)

        landmarks.append([lmx, lmy])

    return landmarks


def predict_gesture(landmarks):
    # how to not print anything
    prediction = np.argmax(model.predict([landmarks], verbose=0))
    className = classNames[prediction]
    historyPredictions.append(className)
    return className


# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            landmarks = landmarks + get_landmarks(handslms)

            # Drawing landmarks on frames
            # mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            className = predict_gesture(landmarks)

            if(className == "stop"):
                if move_cursor(landmarks[8]) is True:
                    continue

            if(len(historyPredictions) > 10):
                do_action(max(set(historyPredictions),
                          key=historyPredictions.count))
                historyPredictions.clear()
    else:
        set_mouse_up()

    # show the prediction on the frame
    # cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
    #             1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
