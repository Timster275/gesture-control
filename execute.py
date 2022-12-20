import json
import pynput.keyboard as keyboard
from pynput.keyboard import Key
class Executor():
    def __init__(self):
        self.entries = json.loads(open('gesture_data/config.json').read())
        self.board = keyboard.Controller()
    def executeSimple(self, gesture):
        for entry in self.entries:
            if entry['gesture'] == gesture:
                print(entry['commands'])
                single_key = entry['commands'].split('+')
                for key in [self.map(k) for k in single_key]:
                    self.board.press(key)
                
                for key in [self.map(k) for k in single_key]:
                    self.board.release(key)

                break
    
    def map(self, key):
        translator = {
            'cmd': Key.cmd_l,
            'ctrl': Key.ctrl,
            'alt': Key.alt,
            'shift': Key.shift,
            'enter': Key.enter,
            'esc': Key.esc,
            'backspace': Key.backspace,
            'tab': Key.tab,
            'caps_lock': Key.caps_lock,
            'space': Key.space,
            'page_up': Key.page_up,
            'page_down': Key.page_down,
            'end': Key.end,
            'home': Key.home,
            'left': Key.left,
            'up': Key.up,
            'right': Key.right,
            'down': Key.down,   
            'delete': Key.delete,
            'f1': Key.f1,
            'f2': Key.f2,
            'f3': Key.f3,
            'f4': Key.f4,
            'f5': Key.f5,
            'f6': Key.f6,
            'f7': Key.f7,
            'f8': Key.f8,
            'f9': Key.f9,
            'f10': Key.f10,
            'f11': Key.f11,
            'f12': Key.f12
        }
        return translator[key]

executor = Executor()
executor.executeSimple('1')