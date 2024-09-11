import numpy as np

with open('/home/ye/Downloads/Robust_LiDAR/LiDAR/carla-nuscenes/dataset/square/training/LIDAR_TOP/n008-2018-08-01-00-00-00-0400__LIDAR_TOP__1500000000000001.pcd.bin','rb') as binary_file:
    binary_data = binary_file.read()

# text_data = binary_data.decode('utf-8')

with open('output.txt', 'w') as text_file:
    # text_file.write(text_data)
    text_file.write(str(binary_data))