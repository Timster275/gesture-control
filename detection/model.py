# get all .dats in the directory
import glob
import json
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 3, subplot_kw=dict(projection='3d'))

files = glob.glob("gesture_data_old/*.dat")
xt = 1
yt = 2
for file in files:
    with open(file, 'r') as f:
        json_data = json.load(f)
        print("Reading " + json_data["name"])
        fx = json_data["x"]
        fy = json_data["y"]
        fz = json_data["z"]
        axs[xt, yt].scatter(fx, fy, fz, label=json_data["name"])
        axs[xt, yt].legend()
        axs[xt, yt].plot(fy, fz)
        yt -= 1
        if yt == -1:
            yt = 2
            xt -= 1
        if xt == -1:
            break

plt.show()

