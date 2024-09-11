import copy
import os
import json
import mmcv
import numpy as np

from nuscenes.nuscenes import NuScenes


from helper import *


def traverse_images(nusc, model_output=None, proj_lidar=False, visible_level=None, save_dir=None):

    if model_output is None:
        # 无算法结果, 默认遍历整个数据集
        for scene in nusc.scene:
            sample = None

            while True:
                if sample is None:
                    sample = nusc.get('sample', scene['first_sample_token'])

                visualize_one_sample(nusc,
                                     sample,
                                     proj_lidar=proj_lidar,
                                     visible_level=visible_level,
                                     scale_ratio=0.45,
                                     save_dir=save_dir)

                if sample['next'] != '':
                    sample = nusc.get('sample', sample['next'])
                else:
                    break


    else:
        # 有算法结果, 按算法结果遍历

        for sample_token in mmcv.track_iter_progress(model_output['results']):
            results = model_output['results'][sample_token]
            sample = nusc.get('sample', sample_token)

            # 可视化数据
            visualize_one_sample(nusc,
                                 sample,
                                 results,
                                 proj_lidar=proj_lidar,
                                 visible_level=visible_level,
                                 save_dir=save_dir)


if __name__ == '__main__':
    # with open('json_file/results_nusc_mini_val.json') as f:
    #     model_outputs = json.load(f)

    data_root = '../carla_nus/dataset/nus/LIDAR_p0_samples'

    nusc = NuScenes(version='v1.0-trainval',
                    dataroot=data_root,
                    verbose=True)
    traverse_images(nusc, model_output=None, proj_lidar=False, visible_level=1, save_dir=None)

    # check_dataset_info(nusc)
