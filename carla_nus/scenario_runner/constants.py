import os
import numpy as np

CAMERA_IMAGE_X = 1600
CAMERA_IMAGE_Y = 900

WINDOW_WIDTH = CAMERA_IMAGE_X
WINDOW_HEIGHT = CAMERA_IMAGE_Y

CAMERA_HEIGHT_POS = 1.50
LIDAR_HEIGHT_POS = CAMERA_HEIGHT_POS

CAMERA_POS_X = 0.0
CAMERA_POS_Y = 0.0
CAMERA_POS_Z = CAMERA_HEIGHT_POS

CAMERA_ROT_PITCH = 0.0
CAMERA_ROT_YAW = 0.0
CAMERA_ROT_ROLL = 0.0

LIDAR_POS_X = 0.0
LIDAR_POS_Y = 0.0
LIDAR_POS_Z = LIDAR_HEIGHT_POS

LIDAR_ROT_PITCH = 0.0
LIDAR_ROT_YAW = 0.0
LIDAR_ROT_ROLL = 0.0

SAVE_DISTANCE = 2
SAVE_INTERVAL = 0.48

MAX_RENDER_DEPTH_IN_METERS = 70.0
MIN_VISIBLE_VERTICES_FOR_RENDER = 4

MIN_VALID_VERTICES_FOR_RENDER = 4
MAX_INVALID_VERTICES_FOR_RENDER = 4

LIMITS = np.array(
    [
        [-66.7, 66.7],
        [-66.7, 66.7],
        [-3.0, 1.0]
    ]
)

MAX_LIMIT = np.array(
    [
        [-66.7, 66.7],
        [-66.7, 66.7],
        [-3.0, 1.0]
    ]
)

# LIMITS = np.array(
#     [
#         [-40.0, 40.0],
#         [-20.0, 20.0],
#         [-3.0, 1.0]
#     ]
# )
#
# MAX_LIMIT = np.array(
#     [
#         [-40.0, 40.0],
#         [-20.0, 20.0],
#         [-3.0, 1.0]
#     ]
# )

MIN_BBOX_AREA_IN_PX = 100

OCCLUDED_VERTEX_COLOR = (255, 0, 0)
VISIBLE_VERTEX_COLOR = (0, 255, 0)

WINDOW_WIDTH_HALF = WINDOW_WIDTH / 2
WINDOW_HEIGHT_HALF = WINDOW_HEIGHT / 2

CLASSES_TO_LABEL = ["Vehicle", "Pedestrian"]

# Save Paths
""" OUTPUT FOLDER GENERATION """

# ignore all the things below
OUTPUT_FOLDER = "XXX/dataset"

""" DATA SAVE PATHS """
# GROUNDPLANE_PATH = os.path.join(OUTPUT_FOLDER, 'training/planes/{0:06}.txt')
# LIDAR_PATH = os.path.join(OUTPUT_FOLDER, 'training/velodyne/{0:06}.bin')
# LIDAR_BEAM_PATH = os.path.join(OUTPUT_FOLDER, 'training/velodyne/{0:06}_{2}.bin')
# LABEL_PATH = os.path.join(OUTPUT_FOLDER, 'training/label_2/{0:06}.txt')
# IMAGE_PATH = os.path.join(OUTPUT_FOLDER, 'training/image_2/{0:06}.jpg')
# CALIBRATION_PATH = os.path.join(OUTPUT_FOLDER, 'training/calib/{0:06}.txt')


# CAMERA CONFIGS

# T_CAM_FRONT = [0.3, 0.0, CAMERA_HEIGHT_POS]
# R_CAM_FRONT = [0.0, 0.0, 0.0]
#
# T_CAM_FRONT_LEFT = [0.2, -0.4, CAMERA_HEIGHT_POS]
# R_CAM_FRONT_LEFT = [0.0, 0.0, -55.0]
#
# T_CAM_FRONT_RIGHT = [0.2, 0.4, CAMERA_HEIGHT_POS]
# R_CAM_FRONT_RIGHT = [0.0, 0.0, 55.0]
#
# T_CAM_BACK = [-1.2, 0.0, CAMERA_HEIGHT_POS]
# R_CAM_BACK = [0.0, 0.0, 180.0]
#
# T_CAM_BACK_LEFT = [-0.4, -0.6, CAMERA_HEIGHT_POS]
# R_CAM_BACK_LEFT = [0.0, 0.0, -110.0]
#
# T_CAM_BACK_RIGHT = [-0.4, 0.6, CAMERA_HEIGHT_POS]
# R_CAM_BACK_RIGHT = [0.0, 0.0, 110.0]


