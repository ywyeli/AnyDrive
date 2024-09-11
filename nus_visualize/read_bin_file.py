##
#

import numpy as np
import os

def label():

    # Replace 'dtype' with the correct data type of your binary file
    data = np.fromfile('./incomplete_echo/moderate/lidarseg/v1.0-trainval/30000000000000000000000000013400_lidarseg.bin',
                       dtype=np.uint8)

    # Reshape if you know the data's shape and it's not a 1D array
    # data = data.reshape((rows, columns))  # Replace with actual dimensions

    print(max(data))

    unique_numbers = set()

    # List to store non-overlapping numbers
    non_overlapping_numbers = []

    for num in data:
        if num not in unique_numbers:
            non_overlapping_numbers.append(num)
            unique_numbers.add(num)

    print(non_overlapping_numbers)

    with open('output.txt', 'a') as f:
        for i in range(len(data)):
            f.write(str(int(data[i])))
            f.write('\n')

# def create():
#
#     # Replace 'dtype' with the correct data type of your binary file
#
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