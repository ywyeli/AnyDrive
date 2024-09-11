import numpy as np
import open3d as o3d
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

# Initialize the nuScenes devkit
nusc = NuScenes(version='v1.0-trainval', dataroot='/media/ye/Data/det_7/wet_ground/moderate', verbose=True)

# Global variables for navigation
current_frame = 0
scene_samples = []


# Function to map labels to colors
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
    # Handle undefined labels by assigning them a default color, e.g., grey
    default_color = [100, 100, 100]
    colors = np.array([color_map.get(label, default_color) for label in labels])
    return colors


# Function to load and prepare a LiDAR point cloud and its segmentation labels for visualization
def load_lidar_pointcloud(sample_token):
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    lidarseg_labels_filename = nusc.get('lidarseg', lidar_token)['filename']

    # Load the point cloud and segmentation labels
    pointcloud = LidarPointCloud.from_file(f'{nusc.dataroot}/{lidar_data["filename"]}')
    labels = np.fromfile(f'{nusc.dataroot}/{lidarseg_labels_filename}', dtype=np.uint8)

    # Map the labels to colors
    colors = map_labels_to_colors(labels)

    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud.points[:3, :].T)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize colors to [0, 1]

    return pcd


# Function to update the visualization based on the current frame
def update_visualization(vis, pcd):
    vis.clear_geometries()
    vis.add_geometry(pcd)


# Initialize scene samples
def init_scene_samples(scene_token):
    global scene_samples
    scene = nusc.get('scene', scene_token)
    first_sample_token = scene['first_sample_token']

    # Collect all sample tokens for the scene
    scene_samples = []
    sample_token = first_sample_token
    while sample_token:
        scene_samples.append(sample_token)
        sample = nusc.get('sample', sample_token)
        sample_token = sample['next']


# Main visualization function with key callbacks
def main_visualization(scene_token):
    global current_frame, scene_samples
    init_scene_samples(scene_token)

    if len(scene_samples) == 0:
        print("No samples found in the scene.")
        return

    # Load the initial point cloud
    pcd = load_lidar_pointcloud(scene_samples[current_frame])

    # Create a visualizer object
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(pcd)

    # Define key callbacks for navigating through frames
    def load_next_frame(vis):
        global current_frame
        current_frame = min(len(scene_samples) - 1, current_frame + 1)
        new_pcd = load_lidar_pointcloud(scene_samples[current_frame])
        update_visualization(vis, new_pcd)

    def load_previous_frame(vis):
        global current_frame
        current_frame = max(0, current_frame - 1)
        new_pcd = load_lidar_pointcloud(scene_samples[current_frame])
        update_visualization(vis, new_pcd)

    # Register key callbacks for the left and right arrow keys
    vis.register_key_callback(262, load_next_frame)  # Right arrow key for next frame
    vis.register_key_callback(263, load_previous_frame)  # Left arrow key for previous frame

    # Run the visualizer
    vis.run()
    vis.destroy_window()

# Example usage: Replace '/path/to/your/dataset' with the actual path to your dataset
scene_token = nusc.scene[280]['token']
main_visualization(scene_token)
