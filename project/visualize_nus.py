import copy

import numpy as np
# import torch
import yaml
import os
import matplotlib.pyplot as plt

try:
    import open3d as o3d
    from open3d import geometry
except ImportError:
    raise ImportError(
        'Please run "pip install open3d" to install open3d first.')

COLOR_MAP = {
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

    # 0: (0, 0, 0),  # noise
    # 1: (112, 128, 144),  # barrier
    # 2: (220, 20, 60),  # bicycle
    # 3: (255,0,0),#(255, 127, 80),  # bus
    # 4: (255, 158, 0),  # car
    # 5:(233, 150, 70),  # construction_vehicle
    # 6:(255, 61, 99),  # motorcycle
    # 7:(0, 0, 230),  # pedestrian
    # 8:(47, 79, 79),  # traffic_cone
    # 9:(255, 140, 0),  # trailer
    # 10:(255, 99, 71),  # Tomato
    # 11: (0, 207, 191),  # nuTonomy green
    # 12:(175, 0, 75),
    # 13:(75, 0, 75),
    # 14:(112, 180, 60),
    # 15:(222, 184, 135),  # Burlywood
    # 16:(0, 175, 0),


    # 0: (0., 0., 0.),
    # 1: (174., 199., 232.),
    # 2: (152., 223., 138.),
    # 3: (31., 119., 180.),
    # 4: (255., 187., 120.),
    # 5: (188., 189., 34.),
    # 6: (140., 86., 75.),
    # 7: (255., 152., 150.),
    # 8: (214., 39., 40.),
    # 9: (197., 176., 213.),
    # 10: (148., 103., 189.),
    # 11: (196., 156., 148.),
    # 12: (23., 190., 207.),
    # 13: (100., 85., 144.),
    # 14: (247., 182., 210.),
    # 15: (66., 188., 102.),
    # 16: (219., 219., 141.),
    # 17: (140., 57., 197.),
    # 18: (202., 185., 52.),
    # 19: (51., 176., 203.),
    # 20: (200., 54., 131.),
    # 21: (92., 193., 61.),
    # 22: (78., 71., 183.),
    # 23: (172., 114., 82.),
    # 24: (255., 127., 14.),
    # 25: (91., 163., 138.),
    # 26: (153., 98., 156.),
    # 27: (140., 153., 101.),
    # 28: (158., 218., 229.),
    # 29: (100., 125., 154.),
    # 30: (178., 127., 135.),
    # 32: (146., 111., 194.),
    # 33: (44., 160., 44.),
    # 34: (112., 128., 144.),
    # 35: (96., 207., 209.),
    # 36: (227., 119., 194.),
    # 37: (213., 92., 176.),
    # 38: (94., 106., 211.),
    # 39: (82., 84., 163.),
    # 40: (100., 85., 144.),
}

def _draw_points(points,
                 vis,
                 points_size=2,
                 point_color=(0.5, 0.5, 0.5),
                 mode='xyzrgb'):
    """Draw points on visualizer.
    Args:
        points (numpy.array | torch.tensor, shape=[N, 3+C]):
            points to visualize.
        vis (:obj:`open3d.visualization.Visualizer`): open3d visualizer.
        points_size (int, optional): the size of points to show on visualizer.
            Default: 2.
        point_color (tuple[float], optional): the color of points.
            Default: (0.5, 0.5, 0.5).
        mode (str, optional):  indicate type of the input points,
            available mode ['xyz', 'xyzrgb']. Default: 'xyz'.
    Returns:
        tuple: points, color of each point.
    """
    vis.get_render_option().point_size = points_size  # set points size
    # if isinstance(points, torch.Tensor):
    #     points = points.cpu().numpy()

    points = points.copy()
    pcd = geometry.PointCloud()
    mode = "xyzrgb"
    if mode == 'xyz':
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        points_colors = np.tile(np.array(point_color), (points.shape[0], 1))
    elif mode == 'xyzrgb':
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        points_colors = points[:, 3:6]
        # normalize to [0, 1] for open3d drawing
        if not ((points_colors >= 0.0) & (points_colors <= 1.0)).all():
            points_colors /= 255.0
    else:
        raise NotImplementedError

    pcd.colors = o3d.utility.Vector3dVector(points_colors)
    vis.add_geometry(pcd)

    o3d.io.write_point_cloud("eg.pcd", pcd)
    return pcd, points_colors


def _draw_bboxes(bbox3d,
                 vis,
                 points_colors,
                 pcd=None,
                 bbox_color=(0, 1, 0),
                 points_in_box_color=(1, 0, 0),
                 rot_axis=2,
                 center_mode='lidar_bottom',
                 mode='xyz'):
    """Draw bbox on visualizer and change the color of points inside bbox3d.
    Args:
        bbox3d (numpy.array | torch.tensor, shape=[M, 7]):
            3d bbox (x, y, z, x_size, y_size, z_size, yaw) to visualize.
        vis (:obj:`open3d.visualization.Visualizer`): open3d visualizer.
        points_colors (numpy.array): color of each points.
        pcd (:obj:`open3d.geometry.PointCloud`, optional): point cloud.
            Default: None.
        bbox_color (tuple[float], optional): the color of bbox.
            Default: (0, 1, 0).
        points_in_box_color (tuple[float], optional):
            the color of points inside bbox3d. Default: (1, 0, 0).
        rot_axis (int, optional): rotation axis of bbox. Default: 2.
        center_mode (bool, optional): indicate the center of bbox is
            bottom center or gravity center. available mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str, optional):  indicate type of the input points,
            available mode ['xyz', 'xyzrgb']. Default: 'xyz'.
    """
    # if isinstance(bbox3d, torch.Tensor):
    #     bbox3d = bbox3d.cpu().numpy()
    bbox3d = bbox3d.copy()

    in_box_color = np.array(points_in_box_color)
    for i in range(len(bbox3d)):
        center = bbox3d[i, 0:3]
        dim = bbox3d[i, 3:6]
        yaw = np.zeros(3)
        yaw[rot_axis] = -bbox3d[i, 6]
        rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)

        if center_mode == 'lidar_bottom':
            center[rot_axis] += dim[
                rot_axis] / 2  # bottom center to gravity center
        elif center_mode == 'camera_bottom':
            center[rot_axis] -= dim[
                rot_axis] / 2  # bottom center to gravity center
        box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)

        line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
        line_set.paint_uniform_color(bbox_color)
        # draw bboxes on visualizer
        vis.add_geometry(line_set)

        # change the color of points which are in box
        if pcd is not None and mode == 'xyz':
            indices = box3d.get_point_indices_within_bounding_box(pcd.points)
            points_colors[indices] = in_box_color

    # update points colors
    if pcd is not None:
        pcd.colors = o3d.utility.Vector3dVector(points_colors)
        vis.update_geometry(pcd)


def show_pts_boxes(points,
                   bbox3d=None,
                   show=True,
                   save_path=None,
                   points_size=2,
                   point_color=(0.5, 0.5, 0.5),
                   bbox_color=(0, 1, 0),
                   points_in_box_color=(1, 0, 0),
                   rot_axis=2,
                   center_mode='lidar_bottom',
                   mode='xyz'):
    """Draw bbox and points on visualizer.
    Args:
        points (numpy.array | torch.tensor, shape=[N, 3+C]):
            points to visualize.
        bbox3d (numpy.array | torch.tensor, shape=[M, 7], optional):
            3D bbox (x, y, z, x_size, y_size, z_size, yaw) to visualize.
            Defaults to None.
        show (bool, optional): whether to show the visualization results.
            Default: True.
        save_path (str, optional): path to save visualized results.
            Default: None.
        points_size (int, optional): the size of points to show on visualizer.
            Default: 2.
        point_color (tuple[float], optional): the color of points.
            Default: (0.5, 0.5, 0.5).
        bbox_color (tuple[float], optional): the color of bbox.
            Default: (0, 1, 0).
        points_in_box_color (tuple[float], optional):
            the color of points which are in bbox3d. Default: (1, 0, 0).
        rot_axis (int, optional): rotation axis of bbox. Default: 2.
        center_mode (bool, optional): indicate the center of bbox is bottom
            center or gravity center. available mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str, optional):  indicate type of the input points, available
            mode ['xyz', 'xyzrgb']. Default: 'xyz'.
    """
    # TODO: support score and class info
    assert 0 <= rot_axis <= 2

    # init visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0])  # create coordinate frame
    vis.add_geometry(mesh_frame)

    # draw points
    pcd, points_colors = _draw_points(points, vis, points_size, point_color,
                                      mode)

    # draw boxes
    if bbox3d is not None:
        _draw_bboxes(bbox3d, vis, points_colors, pcd, bbox_color,
                     points_in_box_color, rot_axis, center_mode, mode)

    if show:
        vis.run()

    if save_path is not None:
        vis.capture_screen_image(save_path)

    vis.destroy_window()


def _draw_bboxes_ind(bbox3d,
                     vis,
                     indices,
                     points_colors,
                     pcd=None,
                     bbox_color=(0, 1, 0),
                     points_in_box_color=(1, 0, 0),
                     rot_axis=2,
                     center_mode='lidar_bottom',
                     mode='xyz'):
    """Draw bbox on visualizer and change the color or points inside bbox3d
    with indices.
    Args:
        bbox3d (numpy.array | torch.tensor, shape=[M, 7]):
            3d bbox (x, y, z, x_size, y_size, z_size, yaw) to visualize.
        vis (:obj:`open3d.visualization.Visualizer`): open3d visualizer.
        indices (numpy.array | torch.tensor, shape=[N, M]):
            indicate which bbox3d that each point lies in.
        points_colors (numpy.array): color of each points.
        pcd (:obj:`open3d.geometry.PointCloud`, optional): point cloud.
            Default: None.
        bbox_color (tuple[float], optional): the color of bbox.
            Default: (0, 1, 0).
        points_in_box_color (tuple[float], optional):
            the color of points which are in bbox3d. Default: (1, 0, 0).
        rot_axis (int, optional): rotation axis of bbox. Default: 2.
        center_mode (bool, optional): indicate the center of bbox is
            bottom center or gravity center. available mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str, optional):  indicate type of the input points,
            available mode ['xyz', 'xyzrgb']. Default: 'xyz'.
    """
    # if isinstance(bbox3d, torch.Tensor):
    #     bbox3d = bbox3d.cpu().numpy()
    # if isinstance(indices, torch.Tensor):
    #     indices = indices.cpu().numpy()
    bbox3d = bbox3d.copy()

    in_box_color = np.array(points_in_box_color)
    for i in range(len(bbox3d)):
        center = bbox3d[i, 0:3]
        dim = bbox3d[i, 3:6]
        yaw = np.zeros(3)
        # TODO: fix problem of current coordinate system
        # dim[0], dim[1] = dim[1], dim[0]  # for current coordinate
        # yaw[rot_axis] = -(bbox3d[i, 6] - 0.5 * np.pi)
        yaw[rot_axis] = -bbox3d[i, 6]
        rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)
        if center_mode == 'lidar_bottom':
            center[rot_axis] += dim[
                rot_axis] / 2  # bottom center to gravity center
        elif center_mode == 'camera_bottom':
            center[rot_axis] -= dim[
                rot_axis] / 2  # bottom center to gravity center
        box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)

        line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
        line_set.paint_uniform_color(bbox_color)
        # draw bboxes on visualizer
        vis.add_geometry(line_set)

        # change the color of points which are in box
        if pcd is not None and mode == 'xyz':
            points_colors[indices[:, i].astype(np.bool)] = in_box_color

    # update points colors
    if pcd is not None:
        pcd.colors = o3d.utility.Vector3dVector(points_colors)
        vis.update_geometry(pcd)


def show_pts_index_boxes(points,
                         bbox3d=None,
                         show=True,
                         indices=None,
                         save_path=None,
                         points_size=2,
                         point_color=(0.5, 0.5, 0.5),
                         bbox_color=(0, 1, 0),
                         points_in_box_color=(1, 0, 0),
                         rot_axis=2,
                         center_mode='lidar_bottom',
                         mode='xyz'):
    """Draw bbox and points on visualizer with indices that indicate which
    bbox3d that each point lies in.
    Args:
        points (numpy.array | torch.tensor, shape=[N, 3+C]):
            points to visualize.
        bbox3d (numpy.array | torch.tensor, shape=[M, 7]):
            3D bbox (x, y, z, x_size, y_size, z_size, yaw) to visualize.
            Defaults to None.
        show (bool, optional): whether to show the visualization results.
            Default: True.
        indices (numpy.array | torch.tensor, shape=[N, M], optional):
            indicate which bbox3d that each point lies in. Default: None.
        save_path (str, optional): path to save visualized results.
            Default: None.
        points_size (int, optional): the size of points to show on visualizer.
            Default: 2.
        point_color (tuple[float], optional): the color of points.
            Default: (0.5, 0.5, 0.5).
        bbox_color (tuple[float], optional): the color of bbox.
            Default: (0, 1, 0).
        points_in_box_color (tuple[float], optional):
            the color of points which are in bbox3d. Default: (1, 0, 0).
        rot_axis (int, optional): rotation axis of bbox. Default: 2.
        center_mode (bool, optional): indicate the center of bbox is
            bottom center or gravity center. available mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str, optional):  indicate type of the input points,
            available mode ['xyz', 'xyzrgb']. Default: 'xyz'.
    """
    # TODO: support score and class info
    assert 0 <= rot_axis <= 2

    # init visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0])  # create coordinate frame
    vis.add_geometry(mesh_frame)

    # draw points
    pcd, points_colors = _draw_points(points, vis, points_size, point_color,
                                      mode)

    # draw boxes
    if bbox3d is not None:
        _draw_bboxes_ind(bbox3d, vis, indices, points_colors, pcd, bbox_color,
                         points_in_box_color, rot_axis, center_mode, mode)

    if show:
        vis.run()

    if save_path is not None:
        vis.capture_screen_image(save_path)

    vis.destroy_window()


class Visualizer(object):
    r"""Online visualizer implemented with Open3d.
    Args:
        points (numpy.array, shape=[N, 3+C]): Points to visualize. The Points
            cloud is in mode of Coord3DMode.DEPTH (please refer to
            core.structures.coord_3d_mode).
        bbox3d (numpy.array, shape=[M, 7], optional): 3D bbox
            (x, y, z, x_size, y_size, z_size, yaw) to visualize.
            The 3D bbox is in mode of Box3DMode.DEPTH with
            gravity_center (please refer to core.structures.box_3d_mode).
            Default: None.
        save_path (str, optional): path to save visualized results.
            Default: None.
        points_size (int, optional): the size of points to show on visualizer.
            Default: 2.
        point_color (tuple[float], optional): the color of points.
            Default: (0.5, 0.5, 0.5).
        bbox_color (tuple[float], optional): the color of bbox.
            Default: (0, 1, 0).
        points_in_box_color (tuple[float], optional):
            the color of points which are in bbox3d. Default: (1, 0, 0).
        rot_axis (int, optional): rotation axis of bbox. Default: 2.
        center_mode (bool, optional): indicate the center of bbox is
            bottom center or gravity center. available mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str, optional):  indicate type of the input points,
            available mode ['xyz', 'xyzrgb']. Default: 'xyz'.
    """

    def __init__(self,
                 points,
                 bbox3d=None,
                 save_path="pic",
                 points_size=2,
                 point_color=(0.5, 0.5, 0.5),
                 bbox_color=(0, 1, 0),
                 points_in_box_color=(1, 0, 0),
                 rot_axis=2,
                 center_mode='lidar_bottom',
                 mode='xyz'):
        super(Visualizer, self).__init__()
        assert 0 <= rot_axis <= 2

        # init visualizer
        self.o3d_visualizer = o3d.visualization.Visualizer()
        self.o3d_visualizer.create_window()
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[0, 0, 0])  # create coordinate frame
        self.o3d_visualizer.add_geometry(mesh_frame)

        self.points_size = points_size
        self.point_color = point_color
        self.bbox_color = bbox_color
        self.points_in_box_color = points_in_box_color
        self.rot_axis = rot_axis
        self.center_mode = center_mode
        self.mode = mode
        self.seg_num = 0

        # draw points
        if points is not None:
            self.pcd, self.points_colors = _draw_points(
                points, self.o3d_visualizer, points_size, point_color, mode)

        # draw boxes
        if bbox3d is not None:
            _draw_bboxes(bbox3d, self.o3d_visualizer, self.points_colors,
                         self.pcd, bbox_color, points_in_box_color, rot_axis,
                         center_mode, mode)

    def add_bboxes(self, bbox3d, bbox_color=None, points_in_box_color=None):
        """Add bounding box to visualizer.
        Args:
            bbox3d (numpy.array, shape=[M, 7]):
                3D bbox (x, y, z, x_size, y_size, z_size, yaw)
                to be visualized. The 3d bbox is in mode of
                Box3DMode.DEPTH with gravity_center (please refer to
                core.structures.box_3d_mode).
            bbox_color (tuple[float]): the color of bbox. Default: None.
            points_in_box_color (tuple[float]): the color of points which
                are in bbox3d. Default: None.
        """
        if bbox_color is None:
            bbox_color = self.bbox_color
        if points_in_box_color is None:
            points_in_box_color = self.points_in_box_color
        _draw_bboxes(bbox3d, self.o3d_visualizer, self.points_colors, self.pcd,
                     bbox_color, points_in_box_color, self.rot_axis,
                     self.center_mode, self.mode)

    def add_seg_mask(self, seg_mask_colors):
        """Add segmentation mask to visualizer via per-point colorization.
        Args:
            seg_mask_colors (numpy.array, shape=[N, 6]):
                The segmentation mask whose first 3 dims are point coordinates
                and last 3 dims are converted colors.
        """
        # we can't draw the colors on existing points
        # in case gt and pred mask would overlap
        # instead we set a large offset along x-axis for each seg mask
        self.seg_num += 1
        offset = (np.array(self.pcd.points).max(0) -
                  np.array(self.pcd.points).min(0))[0] * 1.2 * self.seg_num
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[offset, 0, 0])  # create coordinate frame for seg
        self.o3d_visualizer.add_geometry(mesh_frame)
        seg_points = copy.deepcopy(seg_mask_colors)
        seg_points[:, 0] += offset
        _draw_points(
            seg_points, self.o3d_visualizer, self.points_size, mode='xyzrgb')

    def show(self, save_path=None):
        """Visualize the points cloud.
        Args:
            save_path (str, optional): path to save image. Default: None.
        """

        self.o3d_visualizer.run()

        if save_path is not None:
            self.o3d_visualizer.capture_screen_image(save_path)

        self.o3d_visualizer.destroy_window()
        return


COLOR_MAP = np.array([
    [112, 128, 144],  # barrier
    [220, 20, 60],    # bicycle
    [255, 0, 0],      # bus
    [255, 158, 0],    # car
    [233, 150, 70],   # construction_vehicle
    [255, 61, 99],    # motorcycle
    [0, 0, 230],      # pedestrian
    [47, 79, 79],     # traffic_cone
    [255, 140, 0],    # trailer
    [255, 99, 71],    
    [0, 207, 191],    
    [175, 0, 75],
    [75, 0, 75],
    [112, 180, 60],
    [222, 184, 135],  
    [0, 175, 0],
    [255, 255, 255],  # ignore
]) 


def visualize_semantics(path_pc, path_gt, path_pred, COLOR_MAP):
    
    # load point cloud
    raw_data = np.fromfile(path_pc, dtype=np.float32).reshape((-1, 4))
    points = raw_data[:, 0:3]
    # depth = 1 / np.linalg.norm(raw_data[:, 0:3], 2, axis=1)

    # load semantic label
    gt = np.fromfile(path_gt, dtype=np.uint32).reshape((-1))
    
    # load predictions
    pred = np.fromfile(path_pred, dtype=np.uint32).reshape((-1))
    
    # colorize
    colors = COLOR_MAP[gt]
    print(COLOR_MAP.shape)

    # error map
    correct = (gt == pred) | (gt == 16)
    print(correct.shape)
    error_map = np.ones((gt.shape[0], 3))
    error_map[:, 0] = 231
    error_map[:, 1] = 66
    error_map[:, 2] = 52
    error_map[correct, 0] = 175
    error_map[correct, 1] = 175
    error_map[correct, 2] = 175

    # color_points = np.concatenate((points, colors), axis=1)
    # vis_er = Visualizer(color_points)
    
    error_map_points = np.concatenate((points, error_map), axis=1)
    vis_er = Visualizer(error_map_points)
    
    vis_er.show(save_path="pic.ply")

def visualize_prob(path_pc, Path_prob, colormap=plt.cm.cividis):
    raw_data = np.fromfile(path_pc, dtype=np.float32).reshape((-1, 5))
    points = raw_data[:, 0:3]

    prob_values = np.fromfile(path_prob, dtype=np.float32).reshape((-1))

    gt = np.fromfile(path_gt, dtype=np.uint32)


if __name__ == "__main__":

    # path to point cloud
    path_pc = 'visualize/nus_4995/points_152.bin'
    
    # path to semantic label
    # path_gt = 'visualize/nus_4264/groundtruth_159.label'
    path_gt = path_pc.replace('points', 'groundtruth')
    path_gt = path_gt.replace('bin', 'label')
    
    # path to predictions
    # path_pred = 'visualize/nus_4264/prediction_159.label'
    path_pred = path_pc.replace('points', 'prediction')
    path_pred = path_pred.replace('bin', 'label')
    
    visualize_semantics(path_pc, path_gt, path_pred, COLOR_MAP)

