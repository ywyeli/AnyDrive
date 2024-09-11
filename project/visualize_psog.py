from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
import numpy as np
import open3d as o3d


# Initialize the nuScenes devkit
nusc = NuScenes(version='v1.0-trainval', dataroot='/media/ye/Data/AnyDrive_GT/carla-nuscenes/dataset/trapezoid', verbose=True)


# Function to visualize a single LiDAR point cloud with segmentation labels
def visualize_lidar_segmentation(scene_token):
    # Retrieve the first sample token of the scene
    scene = nusc.get('scene', scene_token)
    sample_token = scene['first_sample_token']
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    lidarseg_labels_filename = nusc.get('lidarseg', lidar_token)['filename']

    # Load point cloud
    pointcloud = LidarPointCloud.from_file(f'{nusc.dataroot}/{lidar_data["filename"]}')

    # Load segmentation labels
    labels = np.fromfile(f'{nusc.dataroot}/{lidarseg_labels_filename}', dtype=np.uint8)

    # Define a function to map labels to colors
    def map_labels_to_colors(labels):
        # Define your label color mapping here
        # This is an example, replace with the actual mapping
        color_map = {
            0: [255, 255, 153],  # "None"
            1: [70, 130, 180],  # "Buildings"
            2: [0, 0, 230],  # "Fences"
            3: [255, 255, 153],  # "Other"
            4: [255, 102, 178],  # "Pedestrians"
            5: [32, 32, 32],  # "Poles"
            6: [224, 224, 224],  # "RoadLines"
            7: [224, 224, 224],  # "Roads"
            8: [0, 204, 204],  # "Sidewalks"
            9: [0, 204, 102],  # "Vegetation"
            10: [210, 105, 30],  # "Vehicles"
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
        colors = np.array([color_map[label] for label in labels])
        return colors

    # Map labels to colors
    colors = map_labels_to_colors(labels)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud.points[:3, :].T)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255)  # Open3D expects colors in [0, 1]

    # Visualize
    o3d.visualization.draw_geometries([pcd])


# Example: Visualize the first scene (replace with a specific scene token)
visualize_lidar_segmentation(nusc.scene[2]['token'])


