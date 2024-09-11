import open3d as o3d
import numpy as np
from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt

# Initialize the NuScenes object from the devkit
nusc = NuScenes(version='v1.0-trainval', dataroot='/home/ye/Projects/data/corruption_data/0/fog/moderate', verbose=True)
# nusc = NuScenes(version='v1.0-trainval', dataroot='/home/ye/Projects/data/corruption_data/6/incomplete_echo/moderate', verbose=True)
# nusc = NuScenes(version='v1.0-trainval', dataroot='/home/ye/Projects/data/corruption-data/corruption0/motion_blur/medium', verbose=True)

# Specify the sample to visualize
my_sample = nusc.sample[12080]

# Load the point cloud
lidar_top_data = nusc.get('sample_data', my_sample['data']['LIDAR_TOP'])
pcl_path = nusc.get_sample_data_path(lidar_top_data['token'])
pc = np.fromfile(pcl_path, dtype=np.float32)
points = pc.reshape((-1, 5))[:, :3]  # x, y, z coordinates

# Load segmentation labels
lidarseg_labels_filename = nusc.dataroot + f"/lidarseg/v1.0-trainval/{lidar_top_data['token']}_lidarseg.bin"
labels = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)

# Create a color map for visualization
max_label = np.max(80)
colors = plt.get_cmap("tab20")(labels / max_label)

# Create Open3D point cloud object
cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(points)
cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

# Visualize
o3d.visualization.draw_geometries([cloud])

