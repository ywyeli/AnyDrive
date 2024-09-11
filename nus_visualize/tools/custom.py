import open3d as o3d
import numpy as np

lidarseg_idx2name_mapping = {
    0: 'noise',
    1: 'animal',
    2: 'human.pedestrian.adult',
    3: 'human.pedestrian.child',
    4: 'human.pedestrian.construction_worker',
    5: 'human.pedestrian.personal_mobility',
    6: 'human.pedestrian.police_officer',
    7: 'human.pedestrian.stroller',
    8: 'human.pedestrian.wheelchair',
    9: 'movable_object.barrier',
    10: 'movable_object.debris',
    11: 'movable_object.pushable_pullable',
    12: 'movable_object.trafficcone',
    13: 'static_object.bicycle_rack',
    14: 'vehicle.bicycle',
    15: 'vehicle.bus.bendy',
    16: 'vehicle.bus.rigid',
    17: 'vehicle.car',
    18: 'vehicle.construction',
    19: 'vehicle.emergency.ambulance',
    20: 'vehicle.emergency.police',
    21: 'vehicle.motorcycle',
    22: 'vehicle.trailer',
    23: 'vehicle.truck',
    24: 'flat.driveable_surface',
    25: 'flat.other',
    26: 'flat.sidewalk',
    27: 'flat.terrain',
    28: 'static.manmade',
    29: 'static.other',
    30: 'static.vegetation',
    31: 'vehicle.ego'
}
map_name_from_general_to_segmentation_class = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'c_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
    'flat.driveable_surface': 'driveable',
    'flat.other': 'other_flat',
    'flat.sidewalk': 'sidewalk',
    'flat.terrain': 'terrain',
    'static.manmade': 'manmade',
    'static.vegetation': 'vegetation',
    'noise': 'ignore',
    'static.other': 'ignore',
    'vehicle.ego': 'ignore'
}

map_name_from_segmentation_class_to_segmentation_index = {
    'ignore': 0,
    'barrier': 1,
    'bicycle': 2,
    'bus': 3,
    'car': 4,
    'c_vehicle': 5,
    'motorcycle': 6,
    'pedestrian': 7,
    'traffic_cone': 8,
    'trailer': 9,
    'truck': 10,
    'driveable': 11,
    'other_flat': 12,
    'sidewalk': 13,
    'terrain': 14,
    'manmade': 15,
    'vegetation': 16
}


def create_ground_to_pred_label_map():
    ground_to_pred_label_map = {i: map_name_from_segmentation_class_to_segmentation_index[
        map_name_from_general_to_segmentation_class[
            lidarseg_idx2name_mapping[i]]] for i in range(
        len(map_name_from_general_to_segmentation_class))}
    return ground_to_pred_label_map

def map_ground_label_to_pred_label_domain(ground_lidarseg):
    if ground_lidarseg is None:
         return ground_lidarseg
    label_map = create_ground_to_pred_label_map()
    return np.vectorize(label_map.__getitem__)(ground_lidarseg)

def custom_draw_geometry_with_rotation(pcd):

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(0, 0.0, 1.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd],
                                                              rotate_view)

if __name__ == '__main__':
    kl= list(map_name_from_general_to_segmentation_class.keys())
    kr= list(map_name_from_segmentation_class_to_segmentation_index)
    print(kl)
    print(kr)
    mp = create_ground_to_pred_label_map()
    for k,v in mp.items():
       print(kl[k],kr[v]) 