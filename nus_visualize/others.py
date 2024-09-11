from nuscenes.nuscenes import NuScenes
import mmcv
import numpy as np


def get_dataset_info1(nusc):
    scene_num = len(nusc.scene)
    sample_num = 0
    ann_num = 0

    for scene in nusc.scene:
        sample = None
        while True:
            if sample is None:
                sample = nusc.get('sample', scene['first_sample_token'])

            sample_num += 1
            ann_num += len(sample['anns'])

            if sample['next'] != '':
                sample = nusc.get('sample', sample['next'])
            else:
                break

    print('====== Start from scene')
    print('Scene Num: %d\nSample Num: %d\nAnnotation Num: %d' % (scene_num, sample_num, ann_num))


def get_dataset_info2(nusc):
    sample_num = len(nusc.sample)
    ann_num = 0

    scene_tokens = set()
    for sample in nusc.sample:
        ann_num += len(sample['anns'])

        scene = nusc.get('scene', sample['scene_token'])
        scene_tokens.add(scene['token'])
    scene_num = len(scene_tokens)

    print('====== Start from sample')
    print('Scene Num: %d\nSample Num: %d\nAnnotation Num: %d' % (scene_num, sample_num, ann_num))


def get_dataset_info3(nusc):
    ann_num = len(nusc.sample_annotation)

    scene_tokens = set()
    sample_tokens = set()
    for ann in nusc.sample_annotation:
        sample = nusc.get('sample', ann['sample_token'])
        sample_tokens.add(sample['token'])

        scene = nusc.get('scene', sample['scene_token'])
        scene_tokens.add(scene['token'])
    scene_num = len(scene_tokens)
    sample_num = len(sample_tokens)

    print('====== Start from annotation')
    print('Scene Num: %d\nSample Num: %d\nAnnotation Num: %d' % (scene_num, sample_num, ann_num))


def check_dataset_info(nusc):
    dataset_info = dict()
    dataset_info['expo_time_bet_adj_cams'] = list()
    dataset_info['max_delta_time_bet_cams'] = list()
    dataset_info['cam_refresh_time'] = list()
    dataset_info['lidar_refresh_time'] = list()
    dataset_info['delta_lidar_cam_time'] = list()

    for scene in mmcv.track_iter_progress(nusc.scene):
        sample = None
        cam_time = list()
        lidar_time = list()
        expo_time = list()
        diff_time = list()
        diff_lidar_time = list()
        while True:
            if sample is None:
                sample = nusc.get('sample', scene['first_sample_token'])

            # 相机
            timestamps = list()
            for key in sample['data']:
                if key.startswith('CAM'):
                    sample_data = nusc.get('sample_data', sample['data'][key])
                    timestamps.append(sample_data['timestamp'] * 0.001)
            timestamps.sort()
            cam_time.append(timestamps[0])
            delta_time = np.mean([timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))])
            expo_time.append(delta_time)
            diff_time.append(timestamps[-1] - timestamps[0])

            # 激光
            sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            lidar_time.append(sample_data['timestamp'] * 0.001)
            diff_lidar_time.append(lidar_time[-1] - timestamps[-1])

            if sample['next'] != '':
                sample = nusc.get('sample', sample['next'])
            else:
                # 一个scene结束, 统计数据
                dataset_info['expo_time_bet_adj_cams'].append(np.mean(expo_time))
                dataset_info['max_delta_time_bet_cams'].append(np.mean(diff_time))
                dataset_info['delta_lidar_cam_time'].append(np.mean(diff_lidar_time))
                dataset_info['cam_refresh_time'].append(np.mean([cam_time[i] - cam_time[i - 1] for i in range(1, len(cam_time))]))
                dataset_info['lidar_refresh_time'].append(np.mean([lidar_time[i] - lidar_time[i - 1] for i in range(1, len(lidar_time))]))
                break

    print('======')
    print('Total Scene  Num: %d\nTotal Sample Num: %d' % (len(nusc.scene), len(nusc.sample)))
    max_len = max([len(key) for key in dataset_info])
    for key in dataset_info:
        val_avg = np.mean(dataset_info[key])
        val_2std = np.percentile(dataset_info[key], 95.24)
        print('[%*s] Avg: %7.2fms  2STD: %7.2fms' % (max_len, key, val_avg, val_2std))


if __name__ == '__main__':
    data_root = r''
    nusc = NuScenes(version='v1.0-mini',
                    dataroot=data_root,
                    verbose=True)
    get_dataset_info1(nusc)
    get_dataset_info2(nusc)
    get_dataset_info3(nusc)
    check_dataset_info(nusc)
