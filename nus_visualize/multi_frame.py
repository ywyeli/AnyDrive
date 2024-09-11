import numpy as np
import open3d as o3d
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from scipy.spatial.transform import Rotation as R
from nuscenes.utils.geometry_utils import transform_matrix
import numpy as np
import open3d as o3d
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from scipy.spatial.transform import Rotation as R

# 初始化NuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot='../../carla_nus/dataset/nus')
# 自动获取所有样本tokens

def load_transformed_point_clouds(nusc, num_frames=5, frames_ahead=0):
    """
    加载并转换指定数量的点云帧，以及每帧之后指定数量的帧。
    """
    point_clouds = []
    ff = 0
    for sample in nusc.sample[ff:ff+num_frames]:
        lidar_token = sample['data']['LIDAR_TOP']
        current_pcl, _ = load_and_transform_pcl(nusc, lidar_token)
        ahead_pcls = [current_pcl]

        # 加载并转换后续的几帧
        for _ in range(frames_ahead):
            sample = nusc.get('sample', sample['next'])
            if sample is None:
                break
            lidar_token = sample['data']['LIDAR_TOP']
            next_pcl, _ = load_and_transform_pcl(nusc, lidar_token)
            ahead_pcls.append(next_pcl)

        point_clouds.append(ahead_pcls)
    return point_clouds


def load_and_transform_pcl(nusc, lidar_token):
    """
    加载单个点云并将其转换到世界坐标系。
    """
    lidar_data = nusc.get('sample_data', lidar_token)
    pcl_path = nusc.get_sample_data_path(lidar_token)
    pcl = LidarPointCloud.from_file(pcl_path)

    # 获取变换矩阵并应用
    cs_record = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', lidar_data['ego_pose_token'])

    # 转换到世界坐标系
    # pcl.rotate(R.from_quat(cs_record['rotation']).as_matrix())
    # pcl.translate(np.array(cs_record['translation']))
    # pcl.rotate(R.from_quat(pose_record['rotation']).as_matrix())
    # pcl.translate(np.array(pose_record['translation']))

    pcl.rotate(R.from_quat([cs_record['rotation'][1], cs_record['rotation'][2], cs_record['rotation'][3],
                            cs_record['rotation'][0]]).as_matrix())
    pcl.translate(np.array(cs_record['translation']))
    pcl.rotate(R.from_quat([pose_record['rotation'][1], pose_record['rotation'][2], pose_record['rotation'][3],
                            pose_record['rotation'][0]]).as_matrix())
    pcl.translate(np.array(pose_record['translation']))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl.points[:3, :].T)

    return pcd, lidar_data


# 初始化NuScenes对象
point_clouds = load_transformed_point_clouds(nusc)

# 初始化Open3D可视化器
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()

current_frame = 0


def show_frame(vis):
    global current_frame
    vis.clear_geometries()
    for pcd in point_clouds[current_frame]:
        vis.add_geometry(pcd)


def next_frame(vis):
    global current_frame
    current_frame = min(current_frame + 1, len(point_clouds) - 1)
    show_frame(vis)


def prev_frame(vis):
    global current_frame
    current_frame = max(current_frame - 1, 0)
    show_frame(vis)


# 绑定键盘回调
vis.register_key_callback(265, next_frame)  # Up arrow key
vis.register_key_callback(264, prev_frame)  # Down arrow key

show_frame(vis)  # 显示第一帧及其后续帧

vis.run()
vis.destroy_window()
