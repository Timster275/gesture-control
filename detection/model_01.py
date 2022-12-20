


import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

import random


#
# def medianFilter(img, k=6):
#   w, h, c = img.shape
#   size = k // 2
#
#   #0 padding process
#   _img = np.zeros((w+2*size, h+2*size, c), dtype=np.float)
#   _img[size:size+w, size:size+h] = img.copy().astype(np.float)
#   dst = _img.copy()
#
#   #Filtering process
#   for x in range(w):
#     for y in range(h):
#       for z in range(c):
#         dst[x+size, y+size, z] = np.median(_img[x:x+k, y:y+k, z])
#
#   dst = dst[size:size+w, size:size+h].astype(np.uint8)
#
#   return dst


# def convolution(oldimage, kernel):
#     # image = Image.fromarray(image, 'RGB')
#     image_h = oldimage.shape[0]
#     image_w = oldimage.shape[1]
#
#     kernel_h = kernel.shape[0]
#     kernel_w = kernel.shape[1]
#
#     if (len(oldimage.shape) == 3):
#         image_pad = np.pad(oldimage, pad_width=(
#             (kernel_h // 2, kernel_h // 2), (kernel_w // 2,
#                                               kernel_w // 2), (0, 0)), mode = 'constant',
#                            constant_values = 0).astype(np.float32)
#     elif (len(oldimage.shape) == 2):
#         image_pad = np.pad(oldimage, pad_width=(
#             (kernel_h // 2, kernel_h // 2), (kernel_w // 2,
#                                               kernel_w // 2)), mode = 'constant', constant_values = 0).astype(np.float32)
#
#
#     h = kernel_h // 2
#     w = kernel_w // 2
#
#     image_conv = np.zeros(image_pad.shape)
#
#     for i in range(h, image_pad.shape[0] - h):
#         for j in range(w, image_pad.shape[1] - w):
#             # sum = 0
#             x = image_pad[i - h:i - h + kernel_h, j - w:j - w + kernel_w]
#             x = x.flatten() * kernel.flatten()
#             image_conv[i][j] = x.sum()
#     h_end = -h
#     w_end = -w
#
#     if (h == 0):
#         return image_conv[h:, w:w_end]
#     if (w == 0):
#         return image_conv[h:h_end, w:]
#
#     return image_conv[h:h_end, w:w_end]


# def GaussianBlurImage(image, sigma = 0.5):
#     # image = imread(image)
#
#     # print(image)
#     filter_size = 2 * int(4 * sigma + 0.5) + 1
#
#     gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
#     m = filter_size // 2
#     n = filter_size // 2
#
#     for x in range(-m, m + 1):
#         for y in range(-n, n + 1):
#             x1 = 2 * np.pi * (sigma ** 2)
#             x2 = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
#             gaussian_filter[x + m, y + n] = (1 / x1) * x2
#
#     im_filtered = np.zeros_like(image, dtype=np.float32)
#     for c in range(3):
#         im_filtered[:, :, c] = convolution(image[:, :, c], gaussian_filter)
#     return (im_filtered.astype(np.uint8))





def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def gausss(image, var=0.01):
    row, col, ch = image.shape
    mean = 0
    # var = 0.01 ##the less the less noise
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = 255 * gauss  # Now scale by 255
    gauss = gauss.astype(np.uint8)
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return noisy




def callback(x):
    pass  # no body needed in that callback function


def add_noise(pixels, img):
    # Getting the dimensions of the image
    row, col, depth = img.shape
    # print(row, col)

    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    # number_of_pixels = 5000
    for i in range(round(pixels/2)):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to white
        img[y_coord][x_coord][0] = 255
        img[y_coord][x_coord][1] = 255
        img[y_coord][x_coord][2] = 255

    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    # number_of_pixels = 5000
    for i in range(round(pixels/2)):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to black
        img[y_coord][x_coord][0] = 0
        img[y_coord][x_coord][1] = 0
        img[y_coord][x_coord][2] = 0

    return img
# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
# model = load_model('mp_hand_gesture')
model = load_model('model.hdf5')

# Load class names
# f = open('gesture.names', 'r')
f = open('gesture.names8', 'r')

classNames = f.read().split('\n')
f.close()
print(classNames)

cv2.namedWindow('trackbar', 2)
cv2.resizeWindow("trackbar", 550, 10);


cv2.createTrackbar('s&p', 'trackbar', 0, 300000, callback)
cv2.createTrackbar('brightness', 'trackbar', 0, 255, callback)
cv2.createTrackbar('gauss', 'trackbar', 0, 100, callback)


cv2.createTrackbar('Median filter', 'trackbar', 0, 1, callback)
cv2.createTrackbar('Gaussian filter', 'trackbar', 0, 1, callback)

# Initialize the webcam
cap = cv2.VideoCapture(0)
i = 0
while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    # x, y, c = frame.shape

    image_width, image_height = frame.shape[1], frame.shape[0]

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)

    number_of_pixels = cv2.getTrackbarPos('s&p', 'trackbar')
    brightness = cv2.getTrackbarPos('brightness', 'trackbar')
    gaus = cv2.getTrackbarPos('gauss', 'trackbar')/1000

    m = cv2.getTrackbarPos('Median filter', 'trackbar')
    g = cv2.getTrackbarPos('Gaussian filter', 'trackbar')

    framergb = change_brightness(frame, value=brightness)
    framergb = add_noise(number_of_pixels, framergb)
    framergb = gausss(framergb, gaus)

    # framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # framergb = frame


    if m == 1:
        framergb = cv2.medianBlur(framergb, 3)

    if g == 1:
        framergb = cv2.GaussianBlur(framergb, (5, 5), 0)

    # Get hand landmark prediction
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
                landmarks.append(lmx)
                landmarks.append(lmy)

                i = i + 1


            mpDraw.draw_landmarks(framergb, handslms, mpHands.HAND_CONNECTIONS)

            landmarks = np.array(landmarks)

            landmarks = landmarks/abs(max(landmarks, key=abs))

            landmarks = landmarks.reshape(42, 1)
            prediction = model.predict(np.array([landmarks]))


            classID = np.argmax(prediction)
            # className = classNames[classID]

            print(classID)
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


    # show the prediction on the frame
    cv2.putText(framergb, className, (450, 220), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0, 0, 255), 2, cv2.LINE_AA)



    # Show the final output
    cv2.imshow("Output", framergb)

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()