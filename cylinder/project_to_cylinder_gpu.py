import numpy as np
import cv2
import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from cylinder_configs import *
import cupy as cp
from numba import cuda, float32, int32
from tqdm import tqdm

import numpy as np
import cv2
import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from cylinder_configs import *
import cupy as cp
from numba import cuda, float32, int32
import math

@cuda.jit
def process_pixels_gpu(cam_intrinsics, cam_extrinsics, cam_heights, images, start_y, end_y, cylinder_width, cylinder_height, cylinder_image):
    x, y = cuda.grid(2)

    if x >= cylinder_width or y >= (end_y - start_y):
        return

    y_global = y + start_y
    theta = - (x / cylinder_width) * 2 * math.pi + math.pi

    if y_global > 240:
        radius = (615.5 / (y_global - cylinder_height / 2)) * 1.5
        point_h = 0
    else:
        radius = 51.29167
        point_h = (cylinder_height / 2 - y_global) * 1.5 / 18 + 1.5

    for i in range(cam_intrinsics.shape[0]):
        cam_intrinsic = cam_intrinsics[i]
        cam_extrinsic = cam_extrinsics[i]
        # cam_height = cam_heights[i]
        img = images[i]

        x_world = radius * math.cos(theta)
        y_world = radius * math.sin(theta)
        z_world = point_h

        world_coords = cuda.local.array((4,), dtype=float32)
        world_coords[0] = x_world
        world_coords[1] = y_world
        world_coords[2] = z_world
        world_coords[3] = 1.0

        # Transform world coordinates to camera coordinates
        camera_coords = cuda.local.array((3,), dtype=float32)
        for j in range(3):
            camera_coords[j] = 0
            for k in range(4):
                camera_coords[j] += world_coords[k] * cam_extrinsic[j, k]

        # Check if the point is in front of the camera
        if camera_coords[2] <= 0:
            continue  # The point is behind the camera and not visible

        # Project camera coordinates to pixel coordinates
        pixel_coords = cuda.local.array((3,), dtype=float32)
        for j in range(3):
            pixel_coords[j] = 0
            for k in range(3):
                pixel_coords[j] += camera_coords[k] * cam_intrinsic[j, k]

        u = int(pixel_coords[0] / pixel_coords[2])
        v = int(pixel_coords[1] / pixel_coords[2])

        # Check if the pixel coordinates are within the image bounds
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            cylinder_image[y, x, 0] = img[v, u, 0]
            cylinder_image[y, x, 1] = img[v, u, 1]
            cylinder_image[y, x, 2] = img[v, u, 2]
            break

def create_cylinder_image(nusc, sample_token, cylinder_height, cylinder_width):
    sample = nusc.get('sample', sample_token)

    camera_data = []
    for cam in camera_names:
        cam_data = nusc.get('sample_data', sample['data'][cam])
        img = cv2.imread(nusc.get_sample_data_path(cam_data['token']))
        cam_intrinsic = np.array(nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])['camera_intrinsic'], dtype=np.float32)
        sensor_data = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        cam_height = sensor_data['translation'][2]
        translation = np.array(sensor_data['translation'], dtype=np.float32)
        rotation = Quaternion(sensor_data['rotation'])
        cam_extrinsic = np.array(transform_matrix(translation, rotation, inverse=True), dtype=np.float32)

        camera_data.append((cam_intrinsic, cam_extrinsic, cam_height, img))

    # Unzip camera_data into individual arrays
    cam_intrinsics, cam_extrinsics, cam_heights, images = zip(*camera_data)
    cam_intrinsics = cp.array(cam_intrinsics)
    cam_extrinsics = cp.array(cam_extrinsics)
    cam_heights = cp.array(cam_heights)
    images = cp.array(images)

    # Allocate memory on the GPU
    cylinder_image = cp.zeros((cylinder_height, cylinder_width, 3), dtype=cp.uint8)

    # Define the grid size and block size for CUDA kernel
    threads_per_block = (16, 16)
    blocks_per_grid_x = int(np.ceil(cylinder_width / threads_per_block[0]))
    blocks_per_grid_y = int(np.ceil(cylinder_height / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    start_y = 0
    end_y = cylinder_height

    process_pixels_gpu[blocks_per_grid, threads_per_block](cam_intrinsics, cam_extrinsics, cam_heights, images, start_y, end_y, cylinder_width, cylinder_height, cylinder_image)

    return cp.asnumpy(cylinder_image)

if __name__ == '__main__':

    # Initialize nuscenes-devkit
    # nusc = NuScenes(version='v1.0-trainval', dataroot='../carla_nus/dataset/nus/LiDAR_p0_samples', verbose=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot='../../L2_nus/LIDAR_p0_samples', verbose=True)

    camera_names = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT']

    # Save the image
    os.makedirs('./images/', exist_ok=True)
    output_dir = './images/'

    for s in tqdm(nusc.sample):
        sample_token = s['token']
        cylinder_image = create_cylinder_image(nusc, sample_token, CYLINDER_HEIGHT, CYLINDER_WIDTH)

        output_path = os.path.join(output_dir, f"{sample_token}.png")
        cv2.imwrite(output_path, cylinder_image)