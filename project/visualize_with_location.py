import numpy as np
import open3d as o3d
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import os

# Initialize the nuScenes devkit
nusc = NuScenes(version='v1.0-trainval', dataroot='/media/ye/Data/AnyDrive_GT/carla-nuscenes/dataset/trapezoid/samples0', verbose=True)

def map_labels_to_colors(labels):
    color_map = {
        0: [255, 255, 153],  # "None"
        1: [70, 130, 180],  # "Buildings"
        2: [0, 0, 230],  # "Fences"
        3: [255, 255, 153],  # "Other"
        4: [255, 102, 178],  # "Pedestrians"
        5: [255, 153, 153],  # "Poles"
        6: [224, 224, 224],  # "RoadLines"
        7: [224, 224, 224],  # "Roads"
        8: [0, 204, 204],  # "Sidewalks"
        9: [0, 204, 102],  # "Vegetation"
        10: [102, 102, 255],  # "Vehicles"
        11: [0, 128, 255],  # "Walls"
        12: [255, 51, 51],  # "TrafficSigns"
        13: [153, 255, 255],  # "Sky"
        14: [255, 204, 255],  # "Ground"
        15: [255, 127, 80],  # "Bridge"
        16: [96, 96, 96],  # "RailTrack"
        17: [96, 96, 96],  # "GuardRail"
        18: [255, 0, 0],  # "TrafficLight"
        19: [0, 153, 153],  # "Static"
        20: [255, 255, 102],  # "Dynamic"
        21: [0, 128, 255],  # "Water"
        22: [102, 102, 0],  # "Terrain"
    }
    default_color = [204, 204, 255]  # Default color for undefined labels
    colors = np.array([color_map.get(label, default_color) for label in labels])
    return colors

def load_lidar_pointcloud(sample_token):
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    lidarseg_labels_filename = nusc.get('lidarseg', lidar_token)['filename']

    # Load the point cloud
    pointcloud = LidarPointCloud.from_file(f'{nusc.dataroot}/{lidar_data["filename"]}')
    labels = np.fromfile(f'{nusc.dataroot}/{lidarseg_labels_filename}', dtype=np.uint8)

    # Map the labels to colors
    colors = map_labels_to_colors(labels)

    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud.points[:3, :].T)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize colors to [0, 1]

    return pcd

# def read_transformations(file_path):
#     transformations = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             parts = line.strip().split(',')
#             translation = np.array(parts[:3], dtype=np.float64)
#             rotation = np.array(parts[3:], dtype=np.float64)  # Assuming quaternion
#             transformations.append((translation, rotation))
#     return transformations


def read_transformations(file_path):
    """
    Read transformations from a file where each line is x, y, z, roll, pitch, yaw.
    """
    transformations = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            translation = np.array(parts[:3], dtype=np.float64)
            euler_angles = np.array(parts[3: ], dtype=np.float64)  # roll, pitch, yaw
            transformations.append((translation, euler_angles))
    return transformations


def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to a rotation matrix.
    """
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx
    return R

# def apply_transformation(pcd, translation, rotation):
#     # Convert quaternion to rotation matrix
#     rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(rotation)
#     pcd.rotate(rotation_matrix)
#     pcd.translate(translation)
#     return pcd

def apply_transformation(pcd, translation, euler_angles):
    """
    Apply translation and rotation (from Euler angles) to the point cloud.
    """
    rotation_matrix = euler_to_rotation_matrix(*euler_angles)
    pcd.rotate(rotation_matrix)
    pcd.translate(translation)
    return pcd

def main_visualization():

    # Global variables for navigation
    current_scene_index = 229
    # Adjust according to your specific scene

    # Load transformations - adjust the file path as needed
    OUTPUT_FOLDER = '/media/ye/Data/AnyDrive_GT/carla-nuscenes/dataset/trapezoid/training/location/'
    LOCATION_PATH = os.path.join(OUTPUT_FOLDER, '{0:06}.txt')

    # Assuming scene_samples have been initialized
    scene = nusc.scene[current_scene_index]
    scene_token = scene['token']
    scene_samples = []
    sample_token = scene['first_sample_token']
    while sample_token:
        scene_samples.append(sample_token)
        sample = nusc.get('sample', sample_token)
        sample_token = sample['next']

    # Load and transform point clouds
    merged_pcd = o3d.geometry.PointCloud()
    for idx in range(20):
        sample_token = scene_samples[idx]
    # for idx, sample_token in enumerate(scene_samples):
        ic = idx + 1 + current_scene_index * 40
        location_filename = LOCATION_PATH.format(ic)
        transformations = read_transformations(location_filename)
        # if idx >= len(transformations):  # Safety check
        #     break
        pcd = load_lidar_pointcloud(sample_token)
        translation, rotation = transformations[0]

        q_translation = [- translation[0], translation[1], translation[2]]
        rotation = [0, 0, - rotation[1]]
        transformed_pcd = apply_transformation(pcd, q_translation, rotation)
        merged_pcd += transformed_pcd

    # Visualize the merged point cloud
    o3d.visualization.draw_geometries([merged_pcd], window_name="Merged Point Clouds")

if __name__ == "__main__":
    main_visualization()
