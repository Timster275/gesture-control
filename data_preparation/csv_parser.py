import csv
import pandas as pd
import glob
import json

with open('gesture_data_new/labels.csv', 'w') as f1:
    files = glob.glob("gesture_data_new/*.dat")
    for file in files:
        print(file)
        with open(file, 'r') as f:
            json_data = json.load(f)
            for element in json_data['x']:
                for element1 in element:
                    f1.write(str(element1) + ',')
                f1.write('\n')
        
