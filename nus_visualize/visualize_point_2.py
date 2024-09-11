# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Environment carla

import numpy as np

import struct
import open3d

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def load_pc_kitti(pc_path):
    print('useless')
    scan = np.fromfile(pc_path, dtype=np.float32)
    scan = scan.reshape((-1, 5))
    points = scan[:, :]  # get xyz

    f = open('', 'w')
    for i in range(points.shape[0]):
        #for j in range(5):
        strNum = str(points[i][2])
        f.write(strNum)
        f.write(' ')
        f.write('\n')
    f.close()

    # print(points)
    return points

def read_bin_velodyne(path):
    # read bin file and transfer to array data
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

def main():
    path = '/media/liye/T7/July3D/carla_nus/dataset/nus/p0_samples/samples/LIDAR_TOP/'
    pcc_path = path + 'n008-2018-08-01-00-00-00-0400__LIDAR_TOP__1500000000000001.pcd.bin'

    # example=read_bin_velodyne(pc_path)
    example = np.fromfile(pcc_path, dtype=np.float32, count=-1).reshape(-1, 5)
    example_xyz=example[:,:3]
    example_xyz=example_xyz[example_xyz[:,2]>-3]

    # From numpy to Open3D
    pcd = open3d.open3d.geometry.PointCloud()
    pcd.points= open3d.open3d.utility.Vector3dVector(example_xyz)
    vis_ = open3d.visualization.Visualizer()
    vis_.create_window()
    vis_.add_geometry(pcd)
    render_options = vis_.get_render_option()
    render_options.point_size = 2
    render_options.background_color = np.array([0, 0, 0])
    vis_.run()
    vis_.destroy_window()
    # pcd.points= open3d.open3d.utility.Vector3dVector(example)
    # open3d.open3d.visualization.draw_geometries([pcd])



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    # load_pc_kitti(pc_path)
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
