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
    if(pyautogui.getActiveWindow() == None):
        return "None"
    return pyautogui.getActiveWindow().title


def do_action(prediction):
    if(prediction != "fist"):
        pyautogui.mouseUp()
    if(prediction == "stop"):
        do_action_open_hand()
    elif(prediction == "fist"):
        do_action_fist()
    elif(prediction == "okay"):
        do_action_okay()


def do_action_open_hand():
    if(settings.isInTaskSwitcher):
        pyautogui.keyDown('right')
        pyautogui.keyUp('right')
        return
    if get_active_window().find("PowerPoint") != -1:
        pyautogui.press('right')
    elif get_active_window().find("Edge") != -1 or get_active_window().find("Chrome") != -1:
        pyautogui.keyDown('ctrl')
        pyautogui.press('tab')
        pyautogui.keyUp('ctrl')
    else:
        pyautogui.press('playpause')


def move_cursor(position):
    if(get_active_window().find("Minecraft") != -1):
        x = position[0]
        y = position[1]
        x = x * screenWidth / 640
        y = y * screenHeight / 480
        pyautogui.moveTo(x, y)
        return True
    return False


def do_action_fist():
    if(settings.isInTaskSwitcher):
        return
    if(get_active_window().find("Minecraft") != -1):
        pyautogui.mouseDown()
        isDown = True
        return
    pyautogui.keyDown('alt')
    pyautogui.press('tab')
    settings.isInTaskSwitcher = True


def do_action_okay():
    if(settings.isInTaskSwitcher == False):
        return
    pyautogui.keyDown('enter')
    pyautogui.keyUp('enter')
    pyautogui.keyUp('alt')
    settings.isInTaskSwitcher = False


def set_mouse_up():
    if(get_active_window().find("Minecraft") != -1 and isDown == True):
        pyautogui.mouseUp()
        return
