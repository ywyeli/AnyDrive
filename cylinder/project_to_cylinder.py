# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import cv2
import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from cylinder_configs import *
import multiprocessing as mp
from tqdm import tqdm


def cylinder_to_world(theta, radius, camera_height, point_h):
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = point_h

    return np.array([x, y, z, 1])


def world_to_pixel(world_coords, camera_intrinsics, camera_extrinsics, image_width, image_height):
    # Transform world coordinates to camera coordinates
    camera_coords = np.dot(world_coords, np.linalg.inv(camera_extrinsics))

    # Check if the point is in front of the camera
    if camera_coords[2] <= 0:
        return None  # The point is behind the camera and not visible

    # Project camera coordinates to pixel coordinates
    pixel_coords = np.dot(camera_intrinsics, camera_coords[:3])
    u = int(pixel_coords[0] / pixel_coords[2])
    v = int(pixel_coords[1] / pixel_coords[2])

    # Check if the pixel coordinates are within the image bounds
    if 0 <= u < image_width and 0 <= v < image_height:
        return u, v
    else:
        return None  # The point is outside the image boundaries and not visible


def process_pixels(camera_data, start_y, end_y, cylinder_width, cylinder_height):
    # cylinder_image
    cylinder_image = np.zeros((end_y - start_y, cylinder_width, 3), dtype=np.uint8)

    for x in range(cylinder_width):
        for y in range(start_y, end_y):
            theta = - (x / cylinder_width) * 2 * np.pi + np.pi

            if y > 240:
                radius = (615.5 / (y - cylinder_height/2)) * 1.5
                point_h = 0
            else:
                radius = 51.29167
                point_h = (cylinder_height/2 - y) * 1.5 / 18 + 1.5

            for cam_intrinsic, cam_extrinsic, cam_height, img in camera_data:
                world_coords = cylinder_to_world(theta, radius, cam_height, point_h)
                pixel_coords = world_to_pixel(world_coords, cam_intrinsic, cam_extrinsic, IMAGE_WIDTH, IMAGE_HEIGHT)

                if pixel_coords is not None:
                    u, v = pixel_coords
                    if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                        cylinder_image[y - start_y, x, :] = img[v, u, :]
                        break

    return cylinder_image


def create_cylinder_image(nusc, sample_token, cylinder_height, cylinder_width):
    sample = nusc.get('sample', sample_token)

    camera_data = []
    for cam in camera_names:
        cam_data = nusc.get('sample_data', sample['data'][cam])
        img = cv2.imread(nusc.get_sample_data_path(cam_data['token']))
        cam_intrinsic = np.array(nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])['camera_intrinsic'])
        sensor_data = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        cam_height = sensor_data['translation'][2]
        translation = np.array(sensor_data['translation'])
        rotation = Quaternion(sensor_data['rotation'])
        cam_extrinsic = transform_matrix(translation, rotation, inverse=True)

        camera_data.append((cam_intrinsic, cam_extrinsic, cam_height, img))

    chunk_size = cylinder_height // mp.cpu_count()
    args = [(camera_data, i * chunk_size, (i + 1) * chunk_size if i != mp.cpu_count() - 1 else cylinder_height,
             cylinder_width, cylinder_height) for i in range(mp.cpu_count())]

    # cylinder_image = np.zeros((cylinder_height, cylinder_width, 3), dtype=np.uint8)

    with mp.Pool(mp.cpu_count()) as pool:
        # results = list(tqdm(pool.starmap(process_pixels, args), total=len(args)))
        results = list(pool.starmap(process_pixels, args))

    cylinder_image = np.vstack(results)

    # for i, result in enumerate(results):
    #     start_y = i * chunk_size
    #     end_y = start_y + result.shape[0]
    #     cylinder_image[start_y:end_y, :, :] = result

    return cylinder_image


if __name__ == '__main__':

    # Initialize nuscenes-devkit
    nusc = NuScenes(version='v1.0-trainval', dataroot='../carla_nus/dataset/nus/LiDAR_p0_samples', verbose=True)

    camera_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

    # Save the image
    os.makedirs('./images/', exist_ok=True)
    output_dir = './images/'

    for s in tqdm(nusc.sample):
        sample_token = s['token']
        cylinder_image = create_cylinder_image(nusc, sample_token, CYLINDER_HEIGHT, CYLINDER_WIDTH)

        output_path = os.path.join(output_dir, f"{sample_token}.png")
        cv2.imwrite(output_path, cylinder_image)

# cv2.imshow('Cylinder View', cylinder_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()