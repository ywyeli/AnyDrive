import open3d as o3d
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

# Initialize nuScenes object
# nusc = NuScenes(version='v1.0-mini', dataroot='/path/to/nuscenes', verbose=True)
nusc = NuScenes(version='v1.0-trainval', dataroot='/media/ye/Data/Multi-LiDAR/carla-nuscenes/dataset/center/LiDAR_p2_samples', verbose=True)

def load_frame_data(sample_token):
    # Get sample data
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    lidar_filepath = nusc.get_sample_data_path(lidar_token)

    # Load point cloud
    pc = np.fromfile(lidar_filepath, dtype=np.float32)
    points = pc.reshape((-1, 5))[:, :3]

    # Load bounding boxes
    _, boxes, _ = nusc.get_sample_data(lidar_token)

    return points, boxes

# def visualize_lidar_and_bboxes(vis, points, bounding_boxes):
#     # Clear previous geometries
#     vis.clear_geometries()
#
#     # Point Cloud
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.paint_uniform_color([0, 0, 1])  # Blue color for points
#
#     vis.add_geometry(pcd)
#
#     # Bounding Boxes
#     for box in bounding_boxes:
#         corners = box.corners().T
#         lines = [[i, j] for i in range(4) for j in range(i+1, 4)]
#         lines += [[i+4, j+4] for i in range(4) for j in range(i+1, 4)]
#         lines += [[i, i+4] for i in range(4)]
#         line_set = o3d.geometry.LineSet(
#             points=o3d.utility.Vector3dVector(corners),
#             lines=o3d.utility.Vector2iVector(lines),
#         )
#         line_set.paint_uniform_color([1, 1, 1])  # White color for bounding boxes
#         vis.add_geometry(line_set)

def visualize_lidar_and_bboxes(vis, points, bounding_boxes):
    # Clear previous geometries
    vis.clear_geometries()

    # Point Cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0, 0, 1])  # Blue color for points
    vis.add_geometry(pcd)

    # Bounding Boxes
    for box in bounding_boxes:
        corners = box.corners().T
        # Define lines without diagonals
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom rectangle
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top rectangle
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical lines
        ]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(corners),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.paint_uniform_color([1, 1, 1])  # White color for bounding boxes
        vis.add_geometry(line_set)

# def change_frame(vis, direction):
#     global current_sample_token
#     global view_control_params
#
#     # Get next or previous sample
#     if direction == 1:
#         current_sample_token = nusc.get('sample', current_sample_token)['next']
#     else:
#         current_sample_token = nusc.get('sample', current_sample_token)['prev']
#
#     if current_sample_token != '':
#         points, bounding_boxes = load_frame_data(current_sample_token)
#         visualize_lidar_and_bboxes(vis, points, bounding_boxes)
#
#         # Set the view control parameters
#         ctrl = vis.get_view_control()
#         ctrl.convert_from_pinhole_camera_parameters(view_control_params)
#     else:
#         print("No more samples in this direction.")


def change_frame(vis, direction):
    global current_sample_token
    global view_control_params

    # Save the current view point
    view_control_params = vis.get_view_control().convert_to_pinhole_camera_parameters()

    # Get next or previous sample
    if direction == 1:
        current_sample_token = nusc.get('sample', current_sample_token)['next']
    else:
        current_sample_token = nusc.get('sample', current_sample_token)['prev']

    if current_sample_token != '':
        points, bounding_boxes = load_frame_data(current_sample_token)
        visualize_lidar_and_bboxes(vis, points, bounding_boxes)

        # Set the view control parameters
        ctrl = vis.get_view_control()
        ctrl.convert_from_pinhole_camera_parameters(view_control_params)
    else:
        scene_id = nusc.scene
        print("No more samples in this direction.")


if __name__ == "__main__":
    # Start with the first sample

    current_sample_token = nusc.sample[0]['token']

    # Create Open3D visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.get_render_option().background_color = np.array([0, 0, 0])  # Black background

    # Load and visualize initial data
    points, bounding_boxes = load_frame_data(current_sample_token)
    visualize_lidar_and_bboxes(vis, points, bounding_boxes)

    # Save the initial view control parameters
    view_control_params = vis.get_view_control().convert_to_pinhole_camera_parameters()

    # Set key callbacks for changing frames
    vis.register_key_callback(258, lambda vis: change_frame(vis, 1))   # Up arrow key for next frame
    vis.register_key_callback(264, lambda vis: change_frame(vis, -1))   # Down arrow key for previous frame

    # Start visualization
    vis.run()
    vis.destroy_window()
