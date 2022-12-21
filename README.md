# gesture-controll
This will be a project that implements gesture controll for your mac/pc using python 

Using google mediapipe we can track different points on a hand, using them to evaluate if the user is showing a gesture.
Furthermore we can use their popular Face-mesh functionality do determine mimic inputs that could possibly lead to an action too.

# How to install:

Make sure you have python 3.9 installed on your device. 

```
pip install -r requirements.txt
```

## OPTIONAL:
If you are using this script on a mac, you will have to install tensorflow_macos too:
```
pip install tensorflow_mac
```

# How to use
The model (model.hdf5) already implements some gestures. If you want to build your own model you need to follow a few steps:
1. Create Training Data
To create training data run test_data_generator.py. \
Important: It exptects 2 parameters. Parameter 1 is the name of the run (e.g. flat_hand_1). The second parameter is the data class.
```
    python data_preparation/test_data_generator.py <name> <classID>
```
2. Parse the data.
For this step, simply run csv_parser.py
3. Train the model
First, move the newly generated .csv file (generated in the data folder) to the detection folder. Now you can run the jupyter notebook step by step. This will create a .hdf5 file. Move this file to the root directory.
