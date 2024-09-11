##
#

import numpy as np
import os

def label():

    # Replace 'dtype' with the correct data type of your binary file
    data = np.fromfile('/home/ye/Downloads/Robust_LiDAR/LiDAR/carla-nuscenes/dataset/square/training/lidarseg/n008-2018-08-01-00-00-00-0400__LIDAR_TOP__1500000000000112_lidarseg.bin',
                       dtype=np.uint8)

    # Reshape if you know the data's shape and it's not a 1D array
    # data = data.reshape((rows, columns))  # Replace with actual dimensions

    with open('output.txt', 'a') as f:
        for i in range(len(data)):
            f.write(str(int(data[i])))
            f.write('\n')

# def create():
#
#     # Replace 'dtype' with the correct data type of your binary file
#     data = np.fromfile('/home/ye/Downloads/lidarseg/v1.0-mini/0ab9ec2730894df2b48df70d0d2e84a9_lidarseg.bin',
#                        dtype=np.uint8)
#
#     # Reshape if you know the data's shape and it's not a 1D array
#     # data = data.reshape((rows, columns))  # Replace with actual dimensions
#
#     with open('output.txt', 'a') as f:
#         for i in range(len(data)):
#             f.write(str(data[i]))
#             f.write('\n')

if __name__ == '__main__':
    label()