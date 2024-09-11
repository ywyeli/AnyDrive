import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import os
from pyquaternion import Quaternion

# 设置NuScenes数据集路径
data_root = './'
nusc = NuScenes(version='v1.0-trainval', dataroot=data_root, verbose=True)

# 输出 lidarseg 数据的保存目录
lidarseg_output_dir = './lidarseg'


# 遍历所有样本
for sample in nusc.sample:

    # 加载激光雷达数据

    path_pcc = 'samples/LIDAR_TOP/n008-2018-08-01-00-00-00-0400__LIDAR_TOP__' + str(sample['timestamp']) + '.pcd.bin'
    # lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    # lidar_pc = LidarPointCloud.from_file(os.path.join(data_root, path_pcc))
    lidar_pc = np.fromfile(os.path.join(data_root, path_pcc), dtype=np.float32, count=-1).reshape(-1, 5)


    # 获取边界框信息
    annotations = nusc.sample_annotation
    lidarseg_data = []
    for i in range(0, lidar_pc.shape[0]):
        lidarseg_data.append(31)
    print(lidar_pc.shape[0])
    lidar_points = lidar_pc[:, :3]

    for annotation in annotations:
        if annotation['sample_token'] == sample['token']:
            translation = annotation['translation']
            rotation = Quaternion(annotation['rotation'])
            size = annotation['size']
            for i in range(0, lidar_pc.shape[0]):

                lx = lidar_points[i][1]
                ly = - lidar_points[i][0]

                lx = lx - translation[0]
                ly = ly - translation[1]

                point = [lx, ly, 0]

                # 将点云坐标系下的点转换到边界框坐标系下
                lidar_points_local = rotation.inverse.rotate(point)

                # 计算边界框的范围，考虑边界框的旋转

                box_min0 = - size[1] /2
                box_min1 = - size[0] /2
                box_min2 = - size[2]/2

                box_max0 = size[1] /2
                box_max1 = size[0] /2
                box_max2 = size[2]/2

                # 使用条件选择符将点分配给类别的整数编码
                # mask = (lidar_points_local[0] >= box_min0) & (lidar_points_local[0] <= box_max0) & \
                #        (lidar_points_local[1] >= box_min1) & (lidar_points_local[1] <= box_max1) & \
                       # (lidar_points_local[2] >= box_min2) & (lidar_points_local[2] <= box_max2)
                if lidar_points_local[0] >= box_min0:
                    if lidar_points_local[0] <= box_max0:
                        if lidar_points_local[1] >= box_min1:
                            if lidar_points_local[1] <= box_max1:
                                lidarseg_data[i] = 27
                # else:
                #     category_index = 31

        # category_index = category_to_index.get(annotation['category_name'], 0)



    lidarseg_data = np.array(lidarseg_data, dtype=np.uint8)

    # with open('output.txt', 'w') as a:
    #     for m in range(0, lidar_pc.shape[0]):
    #         print(lidarseg_data[m], file=a)

    lidarseg_filename = os.path.join(lidarseg_output_dir, f'{sample["timestamp"]}.bin')

    os.makedirs("./lidarseg", exist_ok=True)
    f = open(lidarseg_filename, 'w')
    lidarseg_data.tofile(f)
