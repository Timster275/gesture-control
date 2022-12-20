import cv2
import mediapipe as mp
import pyautogui as pyautogui
from time import sleep
import time as time
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
historyLandmarks = []


def detect_window_in_focus():
    """ 
    Detects the window in focus. 
    """
    # Get the window in focus
    window = pyautogui.getActiveWindow()
    window_name = window.title
    window_class = window._hWnd

    return window_name, window_class


def do_action():
    """
    Do action based on the hand landmarks
    """
    # use history landmarks to detect the movement of the hand

    # if hand moves to the right quickly then do a skip of the song playing in the background
    # you need to check if the window exists in the background
    # if hand moves to the left quickly then do a rewind of the song playing in the background
    # you need to check if the window exists in the background
    # if hand moves up, increase the volume of the song playing in the background
    # you need to check if the window exists in the background
    # if hand moves down, decrease the volume of the song playing in the background
    # you need to check if the window exists in the background
    # if hand is still for a while, pause the song playing in the background
    # you need to check if the window exists in the background
    # if hand is still for a while, play the song playing in the background
    # you need to check if the window exists in the background
    # detect movement of the hand
    # finger_tip_coordinates = []
    # for finger in hand_landmarks.landmark:
    #     finger_tip_coordinates.append(finger)
    # print(finger_tip_coordinates)

    # detect if the hand is still for a while
    # if the hand is still for a while then pause the song playing in the background
    # you need to check if the window exists in the background
    tolerance = 0.05
    if(check_if_still_in_tolerance(tolerance) is True):
        window_name, window_class = detect_window_in_focus()
        print(window_name)
        if(detect_window_in_focus()[0].find("PowerPoint") != -1):
            pyautogui.press('right')
        elif(detect_window_in_focus()[0].find("Edge") != -1 or detect_window_in_focus()[0].find("Chrome") != -1):
            pyautogui.keyDown('ctrl')
            pyautogui.press('tab')
            pyautogui.keyUp('ctrl')
        else:
            pyautogui.press('playpause')
            # alert the user that the song is paused
            pyautogui.alert(text='Song is paused',
                            title='Song Paused', timeout=1000)
            print("pressing playpause")
    else:
        print("is not playing")


def check_if_still_in_tolerance(tolerance):
    """
    Check if the hand is still for a while
    """
    # get the last 30 landmarks
    # check if the hand is still for a while
    # if the hand is still for a while then pause the song playing in the background
    # you need to check if the window exists in the background
    # there should be some tolerance for the movement of the hand
    # if it is in tolerance then the hand is still
    # landmarks are an object with x, y, z coordinates

    for i in range(0, len(historyLandmarks) - 1):
        for j in range(0, len(historyLandmarks[i][0].landmark) - 1):
            if(abs(historyLandmarks[i][0].landmark[j].x - historyLandmarks[i + 1][0].landmark[j].x) > tolerance):
                return False
            if(abs(historyLandmarks[i][0].landmark[j].y - historyLandmarks[i + 1][0].landmark[j].y) > tolerance):
                return False
            if(abs(historyLandmarks[i][0].landmark[j].z - historyLandmarks[i + 1][0].landmark[j].z) > tolerance):
                return False

    return True


# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
    for idx, file in enumerate(IMAGE_FILES):
        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv2.flip(cv2.imread(file), 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print handedness and draw hand landmarks on the image.
        print('Handedness:', results.multi_handedness)
        if not results.multi_hand_landmarks:
            continue
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            print('hand_landmarks:', hand_landmarks)
            print(
                f'Index finger tip coordinates: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
            )
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        cv2.imwrite(
            '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
        # Draw hand world landmarks.
        if not results.multi_hand_world_landmarks:
            continue
        for hand_world_landmarks in results.multi_hand_world_landmarks:
            mp_drawing.plot_landmarks(
                hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        # save landmarks and add a timestamp to the dictionary
        # if hand is not detected then do not save the landmarks
        # if hand is detected and start time was 1 second ago then save the landmarks
        if(results.multi_hand_landmarks):
            # append as list
            historyLandmarks.append(results.multi_hand_landmarks)
            if(len(historyLandmarks) > 9):
                do_action()
                historyLandmarks.clear()
                sleep(2)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image.fill(0)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        # send a do action command every 10 frames
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
