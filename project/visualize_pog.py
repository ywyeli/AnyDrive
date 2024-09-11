import numpy as np
import open3d as o3d
import os


def load_point_cloud_and_labels_bin(point_cloud_path, labels_path):
    # Load point cloud from a .bin file
    point_cloud = np.fromfile(point_cloud_path, dtype=np.float32).reshape(-1, 5) # x, y, z, intensity, ring index
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3]) # Use only x, y, z for Open3D point cloud

    # Load labels from a .bin file, assuming labels are stored as int32
    labels = np.fromfile(labels_path, dtype=np.int32)

    return pcd, labels

def visualize_lidar_segmentation_bin(pcd, labels, category_colors):
    # Map each label to its corresponding color
    colors = np.array([category_colors.get(label, [255, 255, 255]) for label in labels]) # Default unknown labels to white

    # Assign colors to points
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize colors to [0, 1]

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])


def main():
    # Paths to your dataset (customize these paths)
    frame_path = '/media/ye/Data/AnyDrive_GT/carla-nuscenes/dataset/trapezoid/samples/LIDAR_TOP/n008-2018-08-01-00-00-00-0400__LIDAR_TOP__1500000000000001.pcd.bin'
    labels_path = '/media/ye/Data/AnyDrive_GT/carla-nuscenes/dataset/trapezoid/lidarseg/v1.0-trainval/30000000000000000000000000000001_lidarseg.bin'

    # Define your category colors here: label_id -> RGB
    category_colors = {
        0: [0, 0, 0],  # "None": (0, 0, 0),  # Black.
        1: [70, 130, 180],  # "Buildings": (70, 130, 180),  # Steelblue
        2: [0, 0, 230],  # "Fences": (0, 0, 230),  # Blue
        3: [135, 206, 235],  # "Other": (135, 206, 235),  # Skyblue,
        4: [100, 149, 237],  # "Pedestrians": (100, 149, 237),  # Cornflowerblue
        5: [219, 112, 147],  # "Poles": (219, 112, 147),  # Palevioletred
        6: [0, 0, 128],  # "RoadLines": (0, 0, 128),  # Navy,
        7: [240, 128, 128],  # "Roads": (240, 128, 128),  # Lightcoral
        8: [138, 43, 226],  # "Sidewalks": (138, 43, 226),  # Blueviolet
        9: [112, 128, 144],  # "Vegetation": (112, 128, 144),  # Slategrey
        10: [210, 105, 30],  # "Vehicles": (210, 105, 30),  # Chocolate
        11: [105, 105, 105],  # "Walls": (105, 105, 105),  # Dimgrey
        12: [47, 79, 79],  # "TrafficSigns": (47, 79, 79),  # Darkslategrey
        13: [188, 143, 143],  # "Sky": (188, 143, 143),  # Rosybrown
        14: [220, 20, 60],  # "Ground": (220, 20, 60),  # Crimson
        15: [255, 127, 80],  # "Bridge": (255, 127, 80),  # Coral
        16: [255, 69, 0],  # "RailTrack": (255, 69, 0),  # Orangered
        17: [255, 158, 0],  # "GuardRail": (255, 158, 0),  # Orange
        18: [233, 150, 70],  # "TrafficLight": (233, 150, 70),  # Darksalmon
        19: [255, 83, 0],  # "Static": (255, 83, 0),
        20: [255, 215, 0],  # "Dynamic": (255, 215, 0),  # Gold
        21: [255, 61, 99],  # "Water": (255, 61, 99),  # Red
        22: [255, 140, 0],  # "Terrain": (255, 140, 0),  # Darkorange

        # Add more categories and their colors as needed
    }

    # Load point cloud and labels
    pcd, labels = load_point_cloud_and_labels_bin(frame_path, labels_path)

    # Visualize
    visualize_lidar_segmentation_bin(pcd, labels, category_colors)


if __name__ == "__main__":
    main()
