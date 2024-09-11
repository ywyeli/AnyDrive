# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import shutil

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def create_visual_data(source_folder, target_parent_folder, lidarseg_folder):

    # 获取源文件夹中的所有文件
    files = os.listdir(source_folder)
    i = 0
    # 遍历每个文件并将其放入对应的子文件夹
    # for file_name in files:
    for j in range(1, 10000):

        file_name = 'n008-2018-08-01-00-00-00-0400__LIDAR_TOP__' + str(1500000000000000 + j) + '.pcd.bin'
        lidarseg_file_name = str(1500000000000000 + j) + '.bin'

        source_file_path = os.path.join(source_folder, file_name)
        lidarseg_file_path = os.path.join(lidarseg_folder, lidarseg_file_name)

        if os.path.exists(source_file_path):
            target_subfolder = os.path.join(target_parent_folder, 'data' + str(i))
            print(i, j)
            i = i + 1
        else:
            continue

        # 获取文件末尾的6个字符
        file_suffix = 'data' + file_name[-10:-4]
        file_no = file_name[-1:]

        # 创建子文件夹（如果不存在）
        # target_subfolder = os.path.join(target_parent_folder, file_suffix)


        os.makedirs(target_subfolder, exist_ok=True)

        point_file_path = os.path.join(target_subfolder, 'points.bin')
        ground_file_path = os.path.join(target_subfolder, 'ground.bin')
        pred_file_path = os.path.join(target_subfolder, 'pred.bin')

        # 将文件移动到对应的子文件夹
        shutil.copy(source_file_path, point_file_path)
        shutil.copy(lidarseg_file_path, ground_file_path)
        shutil.copy(lidarseg_file_path, pred_file_path)
        # os.rename(point_file_path, )


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    strl = ''

    source_folder = './LIDAR_TOP_' + strl + '/'
    target_parent_folder = './data_' + strl + '/'
    ground_folder = './lidarseg_' + strl + '/'

    create_visual_data(source_folder, target_parent_folder, ground_folder)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
