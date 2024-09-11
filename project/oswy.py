import open3d as o3d
import numpy as np
from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
import os
import subprocess

# Replace 'script.py' with the path to your Python script
# subprocess.Popen(["python3", "/home/ye/Programs/carla15/PythonAPI/examples/generate_traffic.py"])

# os.system('python3 ~/Programs/carla15/PythonAPI/examples/generate_traffic.py')

with open('output_p.txt', 'w') as output, open('error_p.txt', 'w') as error:
    subprocess.Popen(["python3", "/home/ye/Programs/carla15/PythonAPI/examples/generate_traffic.py"], stdout=output, stderr=error)

print(1)
