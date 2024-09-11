import numpy as np
from pyquaternion import Quaternion
import copy
import cv2
import os


LIDAR_MIN_X = -20
LIDAR_MAX_X = 20
LIDAR_MIN_Y = -35
LIDAR_MAX_Y = 35


detection_class = {'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian',
                   'traffic_cone', 'barrier'}


def get_rgb_by_distance(cur_val, min_val=0, max_val=50):
    jet_color_matrix = [0, 0, 0.5625,
                        0, 0, 0.6250,
                        0, 0, 0.6875,
                        0, 0, 0.7500,
                        0, 0, 0.8125,
                        0, 0, 0.8750,
                        0, 0, 0.9375,
                        0, 0, 1,
                        0, 0.0625, 1,
                        0, 0.1250, 1,
                        0, 0.1875, 1,
                        0, 0.2500, 1,
                        0, 0.3125, 1,
                        0, 0.3750, 1,
                        0, 0.4375, 1,
                        0, 0.5000, 1,
                        0, 0.5625, 1,
                        0, 0.6250, 1,
                        0, 0.6875, 1,
                        0, 0.7500, 1,
                        0, 0.8125, 1,
                        0, 0.8750, 1,
                        0, 0.9375, 1,
                        0, 1, 1,
                        0.0625, 1, 0.9375,
                        0.1250, 1, 0.8750,
                        0.1875, 1, 0.8125,
                        0.2500, 1, 0.7500,
                        0.3125, 1, 0.6875,
                        0.3750, 1, 0.6250,
                        0.4375, 1, 0.5625,
                        0.5000, 1, 0.5000,
                        0.5625, 1, 0.4375,
                        0.6250, 1, 0.3750,
                        0.6875, 1, 0.3125,
                        0.7500, 1, 0.2500,
                        0.8125, 1, 0.1875,
                        0.8750, 1, 0.1250,
                        0.9375, 1, 0.0625,
                        1, 1, 0,
                        1, 0.9375, 0,
                        1, 0.8750, 0,
                        1, 0.8125, 0,
                        1, 0.7500, 0,
                        1, 0.6875, 0,
                        1, 0.6250, 0,
                        1, 0.5625, 0,
                        1, 0.5000, 0,
                        1, 0.4375, 0,
                        1, 0.3750, 0,
                        1, 0.3125, 0,
                        1, 0.2500, 0,
                        1, 0.1875, 0,
                        1, 0.1250, 0,
                        1, 0.0625, 0,
                        1, 0, 0,
                        0.9375, 0, 0,
                        0.8750, 0, 0,
                        0.8125, 0, 0,
                        0.7500, 0, 0,
                        0.6875, 0, 0,
                        0.6250, 0, 0,
                        0.5625, 0, 0,
                        0.5000, 0, 0]
    jet_color_matrix = np.reshape(np.asarray(jet_color_matrix) * 255, (64, 3))
    jet_color_matrix = jet_color_matrix.astype(dtype=np.uint8)

    cur_val = np.clip(cur_val, min_val, max_val)
    index = (cur_val - min_val) / max_val * 63
    index = np.round(index).astype(np.int)
    rgb_val = jet_color_matrix[index, :]

    return rgb_val


def get_obj3d_from_annotation(ann, ego_data, calib_data):
    obj_ann = dict()

    # 1. 类别
    obj_type = set(ann['category_name'].split('.')).intersection(detection_class)
    if len(obj_type) == 0:
        return None
    else:
        obj_type = obj_type.pop()

    # 2. 3D框
    # global frame
    center = np.array(ann['translation'])
    orientation = np.array(ann['rotation'])
    # 从global frame转换到ego vehicle frame
    quaternion = Quaternion(ego_data['rotation']).inverse
    center -= np.array(ego_data['translation'])
    center = np.dot(quaternion.rotation_matrix, center)
    orientation = quaternion * orientation
    # 从ego vehicle frame转换到sensor frame
    quaternion = Quaternion(calib_data['rotation']).inverse
    center -= np.array(calib_data['translation'])
    center = np.dot(quaternion.rotation_matrix, center)
    orientation = quaternion * orientation
    # 根据中心点和旋转量生成3D框
    x, y, z = center
    w, l, h = ann['size']
    x_corners = l / 2 * np.array([-1, 1, 1, -1, -1, 1, 1, -1])
    y_corners = w / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    z_corners = h / 2 * np.array([-1, -1, -1, -1, 1, 1, 1, 1])
    # 初始中心为(0, 0, 0)
    box3d = np.vstack((x_corners, y_corners, z_corners))
    # 旋转3D框
    box3d = np.dot(orientation.rotation_matrix, box3d)
    # 平移3D框
    box3d[0, :] = box3d[0, :] + x
    box3d[1, :] = box3d[1, :] + y
    box3d[2, :] = box3d[2, :] + z

    obj_ann['data_type'] = ann['data_type']
    obj_ann['type'] = obj_type
    obj_ann['box'] = box3d

    return obj_ann


def project_obj2image(obj3d_list, intrinsic):
    obj2d_list = list()

    trans_mat = np.eye(4)
    trans_mat[:3, :3] = np.array(intrinsic)

    for obj in obj3d_list:
        # step1: 判断目标是否在图像内(相机坐标系z朝前, x朝右)
        in_front = obj['box'][2, :] > 0.1
        if all(in_front) is False:
            continue

        # step2: 转换到像素坐标系
        points = obj['box']
        points = np.concatenate((points, np.ones((1, points.shape[1]))), axis=0)
        points = np.dot(trans_mat, points)[:3, :]
        points /= points[2, :]

        obj2d = {'data_type': obj['data_type'], 'type': obj['type'], 'box': points}
        obj2d_list.append(obj2d)

    return obj2d_list


def project_lidar2image(nusc, img, lidar_pt_list, lidar_file, camera_data):
    img_h, img_w, _ = img.shape
    points = copy.deepcopy(lidar_pt_list.transpose())

    # step1: lidar frame -> ego frame
    calib_data = nusc.get('calibrated_sensor', lidar_file['calibrated_sensor_token'])
    rot_matrix = Quaternion(calib_data['rotation']).rotation_matrix
    points[:3, :] = np.dot(rot_matrix, points[:3, :])
    for i in range(3):
        points[i, :] += calib_data['translation'][i]

    # step2: ego frame -> global frame
    ego_data = nusc.get('ego_pose', lidar_file['ego_pose_token'])
    rot_matrix = Quaternion(ego_data['rotation']).rotation_matrix
    points[:3, :] = np.dot(rot_matrix, points[:3, :])
    for i in range(3):
        points[i, :] += ego_data['translation'][i]

    # step3: global frame -> ego frame
    ego_data = nusc.get('ego_pose', camera_data['ego_pose_token'])
    for i in range(3):
        points[i, :] -= ego_data['translation'][i]
    rot_matrix = Quaternion(ego_data['rotation']).rotation_matrix.T
    points[:3, :] = np.dot(rot_matrix, points[:3, :])

    # step4: ego frame -> cam frame
    calib_data = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])
    for i in range(3):
        points[i, :] -= calib_data['translation'][i]
    rot_matrix = Quaternion(calib_data['rotation']).rotation_matrix.T
    points[:3, :] = np.dot(rot_matrix, points[:3, :])

    # step5: cam frame -> uv pixel
    visible = points[2, :] > 0.1
    colors = get_rgb_by_distance(points[2, :], min_val=0, max_val=50)
    intrinsic = calib_data['camera_intrinsic']
    trans_mat = np.eye(4)
    trans_mat[:3, :3] = np.array(intrinsic)
    points = np.concatenate((points[:3, :], np.ones((1, points.shape[1]))), axis=0)
    points = np.dot(trans_mat, points)[:3, :]
    points /= points[2, :]
    points = points[:2, :]

    # 过滤相机后方点和超出图像边界的点
    visible = np.logical_and(visible, points[0, :] >= 0)
    visible = np.logical_and(visible, points[0, :] < img_w)
    visible = np.logical_and(visible, points[1, :] >= 0)
    visible = np.logical_and(visible, points[1, :] < img_h)
    points = points[:, np.where(visible == 1)[0]].astype(np.int)
    colors = colors[np.where(visible == 1)[0], :]

    for i in range(points.shape[1]):
        c = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
        cv2.circle(img, center=(points[0, i], points[1, i]), radius=1, color=c, thickness=2)


def plot_lidar2bev(img, lidar_pt_list):
    img_h, img_w, _ = img.shape

    for i in range(lidar_pt_list.shape[0]):
        x, y, z, _ = lidar_pt_list[i, :]
        if z < -1.8:
            continue
        u = int((x - LIDAR_MIN_X) / (LIDAR_MAX_X - LIDAR_MIN_X) * img_w)
        v = int(img_h - (y - LIDAR_MIN_Y) / (LIDAR_MAX_Y - LIDAR_MIN_Y) * img_h)
        if u < 0 or u >= img_w or v < 0 or v >= img_h:
            continue
        img[int(v), int(u), :] = (128, 128, 128)


def plot_annotation_info(lidar_bev, camera_img, obj_list):
    assert (lidar_bev is None and camera_img is not None) \
           or (lidar_bev is not None and camera_img is None)

    for obj in obj_list:
        obj_type = obj['type']
        box = obj['box']

        thickness = [2, 3]
        if obj_type == 'car':
            color = (0, 255, 255)
        elif obj_type in ['truck', 'trailer', 'bus', 'construction_vehicle']:
            color = (64, 128, 255)
        elif obj_type == 'pedestrian':
            color = (0, 255, 0)
        elif obj_type in ['bicycle', 'motorcycle']:
            color = (255, 255, 0)
        else:
            continue

        if obj['data_type'] == 'gt':
            color = (255, 255, 255)
            thickness = [1, 2]

        if lidar_bev is not None:
            # 激光俯视图
            img_h, img_w, _ = lidar_bev.shape
            for i in range(4):
                j = (i + 1) % 4
                u1 = int((box[0, i] - LIDAR_MIN_X) / (LIDAR_MAX_X - LIDAR_MIN_X) * img_w)
                v1 = int(img_h - (box[1, i] - LIDAR_MIN_Y) / (LIDAR_MAX_Y - LIDAR_MIN_Y) * img_h)
                u2 = int((box[0, j] - LIDAR_MIN_X) / (LIDAR_MAX_X - LIDAR_MIN_X) * img_w)
                v2 = int(img_h - (box[1, j] - LIDAR_MIN_Y) / (LIDAR_MAX_Y - LIDAR_MIN_Y) * img_h)

                cv2.line(lidar_bev, (u1, v1), (u2, v2), color, thickness=thickness[0])

        else:
            # 相机图像
            box = box.astype(np.int)
            for i in range(4):
                j = (i + 1) % 4
                # 下底面
                cv2.line(camera_img, (box[0, i], box[1, i]), (box[0, j], box[1, j]), color, thickness=thickness[1])
                # 上底面
                cv2.line(camera_img, (box[0, i + 4], box[1, i + 4]), (box[0, j + 4], box[1, j + 4]), color, thickness=thickness[1])
                # 侧边线
                cv2.line(camera_img, (box[0, i], box[1, i]), (box[0, i + 4], box[1, i + 4]), color, thickness=thickness[1])


def visualize_one_sample(nusc,
                         sample,
                         results=None,
                         proj_lidar=False,
                         visible_level=4,
                         scale_ratio=0.3,
                         save_dir=None):
    data_root = nusc.dataroot

    camera_file = dict()
    # 1. 激光数据
    lidar_file = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    # 2. 相机数据
    for key in sample['data']:
        if key.startswith('CAM'):
            # print(key)
            sample_data = nusc.get('sample_data', sample['data'][key])
            # print(sample_data)
            camera_file[sample_data['channel']] = sample_data
    # for camera_type in ['BACK', 'BACK_LEFT', 'BACK_RIGHT', 'FRONT', 'FRONT_LEFT', 'FRONT_RIGHT']:
    #     print(camera_file['CAM_{}'.format(camera_type)])

    # 3. 标注数据
    anns_info = list()

    if results is not None:
        for token in sample['anns']:
            anns_data = nusc.get('sample_annotation', token)
            anns_data['data_type'] = 'gt'
            anns_info.append(anns_data)
        for res in results:
            if res['detection_score'] < 0.5:
                continue
            res['visibility_token'] = 2
            res['category_name'] = res['detection_name']
            res['data_type'] = 'result'
            anns_info.append(res)
    else:
        for token in sample['anns']:
            anns_data = nusc.get('sample_annotation', token)
            anns_data['data_type'] = 'result'
            anns_info.append(anns_data)

    lidar_path = os.path.join(data_root, lidar_file['filename'])

    lidar_pt_list = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))[:, :4]
    # print(lidar_pt_list)

    # Step1: 拼接6个周视相机
    img_list = list()
    ori_img_size = (1600, 900)
    for camera_type in ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'BACK', 'BACK_LEFT', 'BACK_RIGHT']:
        camera_data = camera_file['CAM_{}'.format(camera_type)]

        # 原始图像
        img_path = os.path.join(data_root, camera_data['filename'])
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        assert ori_img_size == (w, h)
        cv2.putText(img,
                    text=camera_type,
                    org=(50, 80),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=4.0,
                    thickness=3,
                    color=(0, 0, 255))

        if proj_lidar:
            # 激光点投影
            # print(123)
            project_lidar2image(nusc, img, lidar_pt_list, lidar_file, camera_data)
        else:
            # 俯视图3d框-->图像3d框
            calib_data = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])
            ego_data = nusc.get('ego_pose', camera_data['ego_pose_token'])
            obj3d_list = list()
            if anns_info is not None:
                for ann in anns_info:
                    if int(ann['visibility_token']) < visible_level:
                        continue
                    obj = get_obj3d_from_annotation(ann, ego_data, calib_data)
                    if obj is not None:
                        obj3d_list.append(obj)
            obj2d_list = project_obj2image(obj3d_list, calib_data['camera_intrinsic'])
            plot_annotation_info(None, img, obj2d_list)

        img_list.append(img)

    img_set1 = np.concatenate(img_list[:3], axis=0)
    img_set2 = np.concatenate(img_list[3:], axis=0)
    camera_img = np.concatenate((img_set1, img_set2), axis=1)
    del img, img_list, img_set1, img_set2
    img_h, img_w, _ = (np.asarray(camera_img.shape) * scale_ratio)
    camera_img = cv2.resize(camera_img, (int(img_w), int(img_h)))

    # Step2: 绘制激光俯视图
    scale_ratio = (LIDAR_MAX_X - LIDAR_MIN_X) / (LIDAR_MAX_Y - LIDAR_MIN_Y)
    lidar_bev = np.zeros((int(img_h), int(img_h * scale_ratio), 3), dtype=np.uint8)
    plot_lidar2bev(lidar_bev, lidar_pt_list)

    # Step3: 绘制标注数据
    calib_data = nusc.get('calibrated_sensor', lidar_file['calibrated_sensor_token'])
    ego_data = nusc.get('ego_pose', lidar_file['ego_pose_token'])
    obj3d_list = list()
    if anns_info is not None:
        for ann in anns_info:
            if int(ann['visibility_token']) < visible_level:
                continue
            obj = get_obj3d_from_annotation(ann, ego_data, calib_data)
            if obj is not None:
                obj3d_list.append(obj)
    plot_annotation_info(lidar_bev, None, obj3d_list)

    total_img = np.concatenate((camera_img, lidar_bev), axis=1)

    visible_info = nusc.get('visibility', str(visible_level))

    if save_dir is None:
        print(lidar_file['filename'], camera_data['filename'])
        cv2.imshow(visible_info['description'], total_img)
        # cv2.imshow(lidar_file['filename'], total_img)
        cv2.waitKey(0)
    else:
        sample = nusc.get('sample', lidar_file['sample_token'])
        scene = nusc.get('scene', sample['scene_token'])
        save_path = os.path.join(save_dir, scene['name'])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pic_num = len(os.listdir(save_path))
        save_path = os.path.join(save_path, '%s-%04d.jpg' % (scene['name'], pic_num + 1))
        cv2.imwrite(save_path, total_img)

