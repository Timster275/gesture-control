import json

class Entry():
    def __init__(self, gesture, commands):
        self.gesture = gesture
        self.commands = commands
        
    def __str__(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
