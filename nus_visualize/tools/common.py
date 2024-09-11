import os
import open3d
import numpy as np
from random import random
import json

def loadPCD(path):
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    points = points[:, :3]
    return points

def getPointCloud(pcd: np.ndarray, colors=None):
    assert len(pcd.shape) == 2 and pcd.shape[1] == 3, "invalid pcd shape: {}".format(
        pcd.shape)
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(pcd)
    if colors is not None:
        assert len(colors.shape) == 2 and colors.shape[1] == 3, "invalid colors shape: {}".format(
            colors.shape)
        point_cloud.colors = open3d.utility.Vector3dVector(colors)
    return point_cloud

def create_color_map(config_path=None, use_list=[]):
    color_map = {_:[random(), random(), random()] for _ in range(32)}
    if config_path:
        with open(config_path,'r') as f:
            tmp_map = {int(k):v  for k,v in json.load(f).items()}
        for _ in use_list: color_map[_] = tmp_map[_]
        del tmp_map
    return color_map

# # cannot show 2d image with open3d
# def getRGBImage(image: np.ndarray):
#     assert len(image.shape) == 3 and (image.shape[-1]==1 or image.shape[-1] ==3), "invalid image shape: {}".format(
#         image.shape)
#     # print(image)
#     img = open3d.geometry.Image(image)
#     # print(img)
#     return img

if __name__ == "__main__":
    test_pcd = np.random.randn(1000, 3)
    point_cloud = getPointCloud(test_pcd)
    open3d.visualization.draw_geometries([point_cloud])

    # # -----------------
    # test_image = np.random.random((256, 256, 3)).astype(np.float32)
    # test_image = np.ones((224, 224, 3), dtype=np.float32)
    # plt.imshow(test_image)
    # plt.show()
    # image = getRGBImage(test_image)
    # print(image)
    # open3d.visualization.draw_geometries([image], point_show_normal=True)
