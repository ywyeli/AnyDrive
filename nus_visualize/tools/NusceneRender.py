import open3d as o3d
import numpy as np
from .common import getPointCloud, loadPCD, create_color_map
from .custom import map_ground_label_to_pred_label_domain

class NusceneRender:
    def __init__(self,points_path,ground_path,pred_path, label_color_map):
        self.points = loadPCD(points_path)
        self.lidarseg = np.fromfile(ground_path, dtype=np.uint8) if ground_path else None
        self.lidarseg = map_ground_label_to_pred_label_domain(self.lidarseg)
        self.pred_lidarseg = np.fromfile(pred_path, dtype=np.uint8) if pred_path else None
        
        self.label_color_map = label_color_map
        self.wrong_color_map = {0:[1,1,1], 1:[1,0,0]} # 0代表没错

    @staticmethod
    def valid(points):
        if points is None:
            return False
        return True

    @staticmethod
    def common(points,color):
        pcd = getPointCloud(points,color)
        pcd.estimate_normals()
        ## remove this line of code for accelerating the project
        # pcd.orient_normals_consistent_tangent_plane(1)
        return pcd

    def reload_color_map(self,new_color_map):
        self.label_color_map = new_color_map

    def ground_pcd(self):
        if not self.valid(self.lidarseg):
            raise Exception('Error Reading ground truth lidarseg')
        color = np.array([self.label_color_map[int(_)] for _ in self.lidarseg])
        return self.common(self.points,color)

    def pred_pcd(self):
        if not self.valid(self.lidarseg):
            raise Exception('Error Reading ground truth lidarseg')
        color = np.array([self.label_color_map[int(_)] for _ in self.pred_lidarseg])
        return self.common(self.points,color)

    def mistake_pcd(self):
        if not self.valid(self.lidarseg) or not self.valid(self.pred_lidarseg) or self.pred_lidarseg.shape != self.lidarseg.shape:
            raise Exception('showing mistake requires both pred and ground truth lidarseg to be same shape')
        logic = np.logical_and(self.pred_lidarseg!=0, self.lidarseg != self.pred_lidarseg)
        indices = np.argwhere(logic).flatten()
        wrong_lidar = np.zeros((len(self.lidarseg),))
        wrong_lidar[indices] = 1

        color = np.array([self.wrong_color_map[int(_)] for _ in wrong_lidar])
        return self.common(self.points,color)