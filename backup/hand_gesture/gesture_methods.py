from common_methods import load_class_names
from time import sleep
import pyautogui


classNames = load_class_names('gesture.names')
# okay
# peace
# thumbs up
# thumbs down
# call me
# stop
# rock
# live long
# fist
# smile


class state:
    isInTaskSwitcher = False


settings = state()
isDown = False
screenWidth, screenHeight = pyautogui.size()


def get_active_window():
    if (pyautogui.getActiveWindow() == None):
        return "None"
    return pyautogui.getActiveWindow().title


def do_action(prediction, movement_direction):
    if (prediction != "fist"):
        pyautogui.mouseUp()
    if (prediction == "stop" or prediction == "live long"):
        do_action_open_hand(movement_direction)
    elif (prediction == "fist"):
        do_action_fist()
    elif (prediction == "okay"):
        do_action_okay()
    elif (prediction == "peace"):
        do_action_peace()


def do_action_open_hand(direction):
    if (settings.isInTaskSwitcher and direction != ''):
        pyautogui.keyDown(direction)
        pyautogui.keyUp(direction)
        return
    if get_active_window().find("PowerPoint") != -1:
        pyautogui.press('right')
    elif get_active_window().find("Edge") != -1 or get_active_window().find("Chrome") != -1:
        pyautogui.keyDown('ctrl')
        pyautogui.press('tab')
        pyautogui.keyUp('ctrl')
    else:
        pyautogui.press('playpause')


def click():
    pyautogui.click()


def do_action_fist():
    if (settings.isInTaskSwitcher):
        pyautogui.keyUp('alt')
        settings.isInTaskSwitcher = False
        return
    if (get_active_window().find("Minecraft") != -1):
        pyautogui.mouseDown()
        isDown = True
        return
    pyautogui.keyDown('alt')
    pyautogui.press('tab')
    settings.isInTaskSwitcher = True
    sleep(0.5)


def do_action_peace():
    if (get_active_window().find("Minecraft") != -1):
        pyautogui.rightClick()
        return


def do_action_okay():
    pass


def set_mouse_up():
    if (get_active_window().find("Minecraft") != -1 and isDown == True):
        pyautogui.mouseUp()
        return
