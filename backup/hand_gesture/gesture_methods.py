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


def get_active_window():
    return pyautogui.getActiveWindow().title


def do_action(prediction):
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


def do_action_fist():
    if(settings.isInTaskSwitcher):
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
