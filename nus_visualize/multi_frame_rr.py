import numpy as np
import open3d as o3d
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from scipy.spatial.transform import Rotation as R

# Initialize NuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot='../../carla_nus/dataset/nus')

def load_and_transform_pcl(nusc, lidar_token):
    lidar_data = nusc.get('sample_data', lidar_token)
    pcl_path = nusc.get_sample_data_path(lidar_token)
    pcl = LidarPointCloud.from_file(pcl_path)

    # Transformations to the world coordinate system
    cs_record = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    pcl.rotate(R.from_quat([cs_record['rotation'][1], cs_record['rotation'][2], cs_record['rotation'][3], cs_record['rotation'][0]]).as_matrix())
    pcl.translate(np.array(cs_record['translation']))
    pcl.rotate(R.from_quat([pose_record['rotation'][1], pose_record['rotation'][2], pose_record['rotation'][3], pose_record['rotation'][0]]).as_matrix())
    pcl.translate(np.array(pose_record['translation']))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl.points[:3, :].T)
    return pcd

def load_scene_point_clouds(nusc, scene_token):
    scene = nusc.get('scene', scene_token)
    first_sample_token = scene['first_sample_token']
    sample_token = first_sample_token

    point_clouds = []

    while sample_token:
        sample = nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        pcd = load_and_transform_pcl(nusc, lidar_token)
        point_clouds.append(pcd)
        sample_token = sample['next']

    return point_clouds

# Initialize Open3D visualizer with key callbacks
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()

current_scene_index = 32
scenes = nusc.scene
point_clouds = load_scene_point_clouds(nusc, scenes[current_scene_index]['token'])
current_frame = 0

def show_frame():
    global current_frame, point_clouds, vis
    vis.clear_geometries()
    for frame_idx in range(current_frame, min(current_frame + 5, len(point_clouds))):  # Display up to 5 frames simultaneously
        vis.add_geometry(point_clouds[frame_idx])

def next_frame(vis):
    global current_frame, current_scene_index, scenes, point_clouds
    # Increment frame, check if we need to switch scenes
    current_frame += 1
    if current_frame >= len(point_clouds) - 5:  # Adjust for multiple frame display
        current_scene_index += 1
        if current_scene_index >= len(scenes):
            current_scene_index = 0  # Loop back to the first scene
        point_clouds = load_scene_point_clouds(nusc, scenes[current_scene_index]['token'])
        current_frame = 0
    show_frame()

def prev_frame(vis):
    global current_frame, current_scene_index, scenes, point_clouds
    # Decrement frame, check if we need to switch scenes
    current_frame -= 1
    if current_frame < 0:
        current_scene_index -= 1
        if current_scene_index < 0:
            current_scene_index = len(scenes) - 1  # Loop to the last scene
        point_clouds = load_scene_point_clouds(nusc, scenes[current_scene_index]['token'])
        current_frame = max(0, len(point_clouds) - 5)  # Adjust for multiple frame display
    show_frame()

# Bind key callbacks for frame navigation
vis.register_key_callback(262, next_frame)  # Right arrow key for next frame
vis.register_key_callback(263, prev_frame)  # Left arrow key for previous frame

# Initial display
show_frame()

vis.run()
vis.destroy_window()
