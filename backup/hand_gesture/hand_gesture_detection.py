# import necessary packages
import multiprocessing
from gesture_methods import do_action, set_mouse_up, click
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from common_methods import load_class_names
from gesture_methods import settings

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')


# Load class names
classNames = load_class_names('gesture.names')

historyPredictions = []
historyLandmarks = []


def get_landmarks(hands_lms):
    landmarks = []
    for lm in hands_lms.landmark:
        # print(id, lm)
        lmx = int(lm.x * x)
        lmy = int(lm.y * y)

        landmarks.append([lmx, lmy])

    return landmarks


def get_movement_direction(landmarks):
    # should be right, left, up, down
    # get the first landmark
    first = landmarks[-1]
    # get the last landmark
    last = landmarks[0]
    # get the difference between the first and last landmark
    diff = [first[0] - last[0], first[1] - last[1]]
    # get the absolute value of the difference
    abs_diff = [abs(diff[0]), abs(diff[1])]
    # get the max value of the difference
    max_diff = max(abs_diff)
    # get the index of the max value
    max_diff_index = abs_diff.index(max_diff)
    # get the direction of the movement
    direction = diff[max_diff_index]
    movement_name = ''
    if (direction > 5):
        movement_name = 'right'
    elif (direction < -5):
        movement_name = 'left'
    print(direction, movement_name)
    return movement_name


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

            className = predict_gesture(landmarks)

            historyLandmarks.append(landmarks[0])

            if (settings.isInTaskSwitcher and len(historyPredictions) > 3):
                do_action(max(set(historyPredictions),
                              key=historyPredictions.count), get_movement_direction(historyLandmarks))
                historyPredictions.clear()
                historyLandmarks.clear()
            if (len(historyPredictions) > 7):
                do_action(max(set(historyPredictions),
                              key=historyPredictions.count), get_movement_direction(historyLandmarks))
                historyPredictions.clear()
                historyLandmarks.clear()
    else:
        set_mouse_up()
        historyPredictions.clear()
        historyLandmarks.clear()

    # show the prediction on the frame
    # cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
    #             1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the final output
    # cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
