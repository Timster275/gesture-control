import PySimpleGUI as sg
import os
import json
from entry import Entry

class Window():
    entries = []
    def __init__(self):
        self.layout = [
            [sg.Text('Config Builder')],
            [
                [sg.Table(values=[],size=(100, 50), headings=['Gesture', 'Commands'], key='listview', enable_events=True, justification='left', num_rows=10, expand_x=True, expand_y=True,  row_height=50, font=("Helvetica", 22))],
                [sg.Button('Add', key='add', font=('Helvetica', 12)), sg.Button('Remove', key='remove', font=('Helvetica', 12)), sg.Button('Save', key='save', font=('Helvetica', 12)), sg.Button('Load', key='load', font=('Helvetica', 12))]
            ],
        ]
        sg.theme('DarkAmber')
        self.window = sg.Window('Window Title', self.layout, resizable=True, finalize=True, size=(1200,600))
    
    def run(self):
        while True:
            event, values = self.window.read()
            if event == sg.WIN_CLOSED:
                break
            if event == 'add':
                window2 = sg.Window('Window Title', [[sg.Text('Gesture'), sg.Input(key='gesture')], [sg.Text('Commands'), sg.Input(key='commands')], [sg.Button('Ok', key='Ok')]])
                while True:
                    event2, values2 = window2.read()
                    if event2 == sg.WIN_CLOSED:
                        break
                    if event2 == 'Ok':
                        self.entries.append(Entry(values2['gesture'], values2['commands']))
                        self.window['listview'].update(values=self.window['listview'].get() + [[values2['gesture'], values2['commands']]])
                        break
                window2.close()
            if event == 'remove':
                self.window['listview'].update(values=[])
            if event == 'save':
                with open('config.json', 'w') as f:
                    f.write(json.dumps(self.entries, default=lambda o: o.__dict__, sort_keys=True, indent=4))
            
        self.window.close()
        del self

window = Window()
window.run()