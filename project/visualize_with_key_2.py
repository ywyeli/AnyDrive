import numpy as np
import open3d as o3d
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import argparse

parser = argparse.ArgumentParser(description="Process specific scenes from NuScenes dataset")

parser.add_argument("--data_root",
                        type=str,
                        help="Path to the file containing scene names to process",
                        default='pkl/voxel_map_4_16_5000.pkl')

args = parser.parse_args()

# Initialize the nuScenes devkit
nusc = NuScenes(version='v1.0-trainval', dataroot=args.data_root, verbose=True)

# Global variables for navigation
current_frame = 0
current_scene_index = 280
scene_samples = []

# Function to map labels to colors
def map_labels_to_colors(labels):
    # Simple example color mapping
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
    default_color = [204, 204, 255]  # Red for undefined labels
    colors = np.array([color_map.get(label, default_color) for label in labels])
    return colors

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

def update_visualization(vis, pcd):
    vis.clear_geometries()
    vis.add_geometry(pcd)

def init_scene_samples(scene_token):
    global scene_samples
    scene_samples = []
    sample_token = scene_token
    while sample_token:
        scene_samples.append(sample_token)
        sample = nusc.get('sample', sample_token)
        sample_token = sample['next']

def load_scene_by_index(scene_index):
    global current_scene_index, current_frame, nusc
    if scene_index < 0 or scene_index >= len(nusc.scene):
        print("Scene index out of range.")
        return None
    current_scene_index = scene_index
    scene = nusc.scene[current_scene_index]
    init_scene_samples(scene['first_sample_token'])
    current_frame = 0  # Reset frame index for the new scene
    if len(scene_samples) > 0:
        return load_lidar_pointcloud(scene_samples[current_frame])
    else:
        print("No samples found in the scene.")
        return None

def main_visualization(scene_index):
    global current_frame
    pcd = load_scene_by_index(scene_index)
    if not pcd:
        return  # Exit if the scene has no samples or is out of range

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(pcd)

    def load_next_frame(vis):
        global current_frame, current_scene_index
        current_frame += 1
        if current_frame >= len(scene_samples):
            # Load the next scene
            new_pcd = load_scene_by_index(current_scene_index + 1)
            if new_pcd:
                update_visualization(vis, new_pcd)
        else:
            new_pcd = load_lidar_pointcloud(scene_samples[current_frame])
            update_visualization(vis, new_pcd)

    def load_previous_frame(vis):
        global current_frame
        current_frame = max(0, current_frame - 1)
        new_pcd = load_lidar_pointcloud(scene_samples[current_frame])
        update_visualization(vis, new_pcd)

    vis.register_key_callback(258, load_next_frame)  # Right arrow key for next frame
    vis.register_key_callback(263, load_previous_frame)  # Left arrow key for previous frame

    vis.run()
    vis.destroy_window()

def main_visualization_v(scene_index):
    global current_frame, scene_samples
    pcd = load_scene_by_index(scene_index)
    if not pcd:
        return  # Exit if the scene has no samples or is out of range

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(pcd)

    def get_viewpoint(vis):
        # Capture the current viewpoint settings
        ctr = vis.get_view_control()
        return ctr.convert_to_pinhole_camera_parameters()

    def set_viewpoint(vis, cam_params):
        # Apply the captured viewpoint settings
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(cam_params)

    def load_next_frame(vis):
        global current_frame, current_scene_index
        cam_params = get_viewpoint(vis)  # Capture current viewpoint
        current_frame += 1
        print(current_scene_index, current_frame)
        if current_frame >= len(scene_samples):
            new_pcd = load_scene_by_index(current_scene_index + 1)
            if new_pcd:
                vis.clear_geometries()
                vis.add_geometry(new_pcd)
        else:
            new_pcd = load_lidar_pointcloud(scene_samples[current_frame])
            vis.clear_geometries()
            vis.add_geometry(new_pcd)
        set_viewpoint(vis, cam_params)  # Apply the captured viewpoint to the new frame

    def load_previous_frame(vis):
        global current_frame, current_scene_index
        cam_params = get_viewpoint(vis)  # Capture current viewpoint
        current_frame = max(0, current_frame - 1)
        print(current_scene_index, current_frame)
        new_pcd = load_lidar_pointcloud(scene_samples[current_frame])
        vis.clear_geometries()
        vis.add_geometry(new_pcd)
        set_viewpoint(vis, cam_params)  # Apply the captured viewpoint to the new frame

    vis.register_key_callback(258, load_next_frame)  # Right arrow key for next frame
    vis.register_key_callback(263, load_previous_frame)  # Left arrow key for previous frame

    vis.run()
    vis.destroy_window()

# Replace '/path/to/your/dataset' with the actual path to your dataset
# Example usage: Automatically advance through scenes
# scene_token = nusc.scene[0]['token']
# main_visualization(scene_token)
main_visualization_v(current_scene_index)
