from nuscenes.nuscenes import NuScenes, NuScenesExplorer
import os, sys, copy, argparse
import numpy as np
from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud
import pickle
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points, BoxVisibility, box_in_image
import multiprocessing as mp
import math
import nuscenes
from PIL import Image

sys.path.insert(0, 'src')
import data_utils

MAX_SCENES = 850


'''
Output filepaths
'''
TRAIN_REF_DIRPATH = os.path.join('training', 'nuscenes')
VAL_REF_DIRPATH = os.path.join('validation', 'nuscenes')
TEST_REF_DIRPATH = os.path.join('testing', 'nuscenes')


TEST_RADAR_NEW_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'nuscenes_test_radar_new.txt')
TEST_RADAR_REPROJECTED_NEW_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'nuscenes_test_radar_reprojected_new.txt')
TEST_BOX_POS_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'nuscenes_test_box_pos.txt')

'''
Set up input arguments
'''
parser = argparse.ArgumentParser()

parser.add_argument('--nuscenes_data_root_dirpath',
    type=str, required=True, help='Path to nuscenes dataset')
parser.add_argument('--nuscenes_data_derived_dirpath',
    type=str, required=True, help='Path to derived dataset')
parser.add_argument('--n_scenes_to_process',
    type=int, default=MAX_SCENES, help='Number of scenes to process')
parser.add_argument('--n_forward_frames_to_reproject',
    type=int, default=12, help='Number of forward frames to project onto a target frame')
parser.add_argument('--n_backward_frames_to_reproject',
    type=int, default=12, help='Number of backward frames to project onto a target frame')
parser.add_argument('--paths_only',
    action='store_true', help='If set, then only produce paths')
parser.add_argument('--n_thread',
    type=int, default=40, help='Number of threads to use in parallel pool')
parser.add_argument('--debug',
    action='store_true', help='If set, then enter debug mode')


args = parser.parse_args()


# Create global nuScene object
nusc = NuScenes(
    version='v1.0-test',
    dataroot=args.nuscenes_data_root_dirpath,
    verbose=True)

nusc_explorer = NuScenesExplorer(nusc)


def _get_class_label_mapping(category_names, category_mapping):
    """
    :param category_mapping: [dict] Map from original name to target name. Subsets of names are supported. 
        e.g. {'pedestrian' : 'pedestrian'} will map all pedestrian types to the same label

    :returns: 
        [0]: [dict of (str, int)] mapping from category name to the corresponding index-number
        [1]: [dict of (int, str)] mapping from index number to category name
    """
    # Initialize local variables
    original_name_to_label = {}
    original_category_names = category_names.copy()

    original_category_names.append('bg')
    if category_mapping is None:
        # Create identity mapping and ignore no class
        category_mapping = dict()
        for cat_name in category_names:
            category_mapping[cat_name] = cat_name

    # List of unique class_names
    selected_category_names = set(category_mapping.values()) # unordered
    selected_category_names = list(selected_category_names)
    selected_category_names.sort() # ordered

    # Create the label to class_name mapping
    label_to_name = { label:name for label, name in enumerate(selected_category_names)}
    label_to_name[len(label_to_name)] = 'bg' # Add the background class

    # Create original class name to label mapping
    for label, label_name in label_to_name.items():

        # Looking for all the original names that are adressed by label name
        targets = [original_name for original_name in original_category_names if label_name in original_name]

        # Assigning the same label for all adressed targets
        for target in targets:

            # Check for ambiguity
            assert target not in original_name_to_label.keys(), 'ambigous mapping found for (%s->%s)'%(target, label_name)

            # Assign label to original name
            # Some label_names will have the same label, which is totally fine
            original_name_to_label[target] = label

    # Check for correctness
    actual_labels = original_name_to_label.values()
    expected_labels = range(0, max(actual_labels)+1) # we want to start labels at 0
    assert all([label in actual_labels for label in expected_labels]), 'Expected labels do not match actual labels'

    return original_name_to_label, label_to_name

def get_train_val_split_ids(debug=False):
    '''
    Given the nuscenes object, find out which scene ids correspond to which set.
    The split is taken from the official nuScene split available here:
    https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/splits.py

    Arg(s):
        debug : bool
            if set, then enter debug mode

    Returns:
        list[int] : list containing ids of the scenes that are training split
        list[int] : list containing ids of the scenes that are validation split
    '''

    train_file_name = os.path.join('data_split', 'train_ids.pkl')
    val_file_name = os.path.join('data_split', 'val_ids.pkl')

    open_file = open(train_file_name, "rb")
    train_ids = pickle.load(open_file)
    open_file.close()

    open_file = open(val_file_name, "rb")
    val_ids = pickle.load(open_file)
    open_file.close()

    if debug:
        train_ids_final = [1]
        return train_ids_final, val_ids

    return train_ids, val_ids

def point_cloud_to_image(nusc,
                         point_cloud,
                         lidar_sensor_token,
                         camera_token,
                         min_distance_from_camera=1.0,
                         include_feature=False):
    '''
    Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
    plane.

    Arg(s):
        nusc : Object
            nuScenes data object
        point_cloud : PointCloud
            nuScenes point cloud object
        lidar_sensor_token : str
            token to access lidar data in nuscenes sample_data object
        camera_token : str
            token to access camera data in nuscenes sample_data object
        minimum_distance_from_camera : float32
            threshold for removing points that exceeds minimum distance from camera
    Returns:
        numpy[float32] : 3 x N array of x, y, z
        numpy[float32] : N array of z
        numpy[float32] : camera image
    '''
    if isinstance(point_cloud, np.ndarray):
        point_cloud = RadarPointCloud(point_cloud)

    # Get dictionary of containing path to image, pose, etc.
    camera = nusc.get('sample_data', camera_token)
    lidar_sensor = nusc.get('sample_data', lidar_sensor_token)

    image_path = os.path.join(nusc.dataroot, camera['filename'])
    image = data_utils.load_image(image_path)

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    pose_lidar_to_body = nusc.get('calibrated_sensor', lidar_sensor['calibrated_sensor_token'])
    point_cloud.rotate(Quaternion(pose_lidar_to_body['rotation']).rotation_matrix)
    point_cloud.translate(np.array(pose_lidar_to_body['translation']))

    # Second step: transform from ego to the global frame.
    pose_body_to_global = nusc.get('ego_pose', lidar_sensor['ego_pose_token'])
    point_cloud.rotate(Quaternion(pose_body_to_global['rotation']).rotation_matrix)
    point_cloud.translate(np.array(pose_body_to_global['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    pose_body_to_global = nusc.get('ego_pose', camera['ego_pose_token'])
    point_cloud.translate(-np.array(pose_body_to_global['translation']))
    point_cloud.rotate(Quaternion(pose_body_to_global['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    pose_body_to_camera = nusc.get('calibrated_sensor', camera['calibrated_sensor_token'])
    point_cloud.translate(-np.array(pose_body_to_camera['translation']))
    point_cloud.rotate(Quaternion(pose_body_to_camera['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depth = point_cloud.points[2, :]

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    # Points will be 3 x N
    points = view_points(point_cloud.points[:3, :], np.array(pose_body_to_camera['camera_intrinsic']), normalize=True)

    # also return other radar features
    if include_feature:
        points = np.concatenate([points, point_cloud.points[3:, :]], axis=0)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depth.shape[0], dtype=bool)
    mask = np.logical_and(mask, depth > min_distance_from_camera)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < image.shape[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < image.shape[0] - 1)

    # Select points that are more than min distance from camera and not on edge of image
    points = points[:, mask]
    depth = depth[mask]

    return points, depth, image

def bbox_convert_center(box):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    ctr_x = x1 + 0.5*width
    ctr_y = y1 + 0.5*height
    return np.array([ctr_x, ctr_y])

def calc_mask_mod(nusc, nusc_sample_data_rad, nusc_sample_data_cam, points3d, category_selection, classes,\
                image_min_side, image_max_side, tolerance=0.0, angle_tolerance=0.0, use_points_in_box2=False):
    """
    :param points3d: <np array of channels x samples]>
    :param category_selection: list of categories, which will be masked
    :param tolerance: cartesian tolerance in meters
    :param angle_tolerances: angular tolerance in rad
    """

    # Create Boxes:
    # _, boxes, camera_intrinsic = nusc.get_sample_data(nusc_sample_data['token'], box_vis_level=nuscenes.utils.geometry_utils.BoxVisibility.ANY)
    box_center = []
    boxes = nusc.get_boxes(nusc_sample_data_rad['token'])
    cs_record = nusc.get('calibrated_sensor', nusc_sample_data_rad['calibrated_sensor_token'])
    cs_record_cam = nusc.get('calibrated_sensor', nusc_sample_data_cam['calibrated_sensor_token'])
    img_size = (nusc_sample_data_cam['width'], nusc_sample_data_cam['height'])
    cam_intrinsic = np.array(cs_record_cam['camera_intrinsic'])

    pose_record = nusc.get('ego_pose', nusc_sample_data_rad['ego_pose_token'])
    pose_record_cam = nusc.get('ego_pose', nusc_sample_data_cam['ego_pose_token'])
    
    bbox_resize = [ 1. / nusc_sample_data_cam['height'], 1. / nusc_sample_data_cam['width'] ]
    bbox_resize[0] *= float(image_min_side)
    bbox_resize[1] *= float(image_max_side)
    box_mask_all = np.zeros((image_min_side, image_max_side, 35))
    
    box_radar_align = np.zeros(points3d.shape[-1])
    mask = np.zeros(points3d.shape[-1])
    radar_proj_height = np.zeros(points3d.shape[-1])
    box_count = 1
    
    pose_record_cam_inv_rot = Quaternion(pose_record_cam['rotation']).inverse
    pose_record_inv_rot = Quaternion(pose_record['rotation']).inverse
    cs_record_cam_inv_rot = Quaternion(cs_record_cam['rotation']).inverse
    cs_record_inv_rot = Quaternion(cs_record['rotation']).inverse

    box_pos = []
    box_dist = []
    box_height = []
    box_height_map = np.zeros((image_min_side, image_max_side))
    
    for idx, box in enumerate(boxes):
        if category_selection is None or box.name in category_selection:
            box_label = classes[box.name] + 1

            box_cam = box.copy()
            box_mask = np.zeros((image_min_side, image_max_side))
            
            ##### Transform with respect to current radar and camera sensor #####
            # Move box to ego vehicle coord system
            box_cam.translate(-np.array(pose_record_cam['translation']))
            box_cam.rotate(pose_record_cam_inv_rot)

            #  Move box to sensor coord system
            box_cam.translate(-np.array(cs_record_cam['translation']))
            box_cam.rotate(cs_record_cam_inv_rot)
            
            box.translate(-np.array(pose_record['translation']))
            box.rotate(pose_record_inv_rot)

            #  Move box to sensor coord system
            box.translate(-np.array(cs_record['translation']))
            box.rotate(cs_record_inv_rot)
            
            ##### Check if this box is visiable in this camera #####
            if box_in_image(box=box_cam, intrinsic=cam_intrinsic, imsize=img_size, vis_level=BoxVisibility.ANY):
                box_dist.append(box.center[0])
                box_height.append(box.wlh[2])

                ##### Check if points are inside box #####
                if use_points_in_box2:
                    cur_mask = nuscenes.utils.geometry_utils.points_in_box2(box, points3d, wlh_tolerance=tolerance, angle_tolerance=angle_tolerance)
                else:
                    cur_mask = nuscenes.utils.geometry_utils.points_in_box(box, points3d)
                
                ##### Generate box mask #####
                box2d = box_cam.box2d(cam_intrinsic)
                box2d = box2d * np.tile(bbox_resize[::-1], 2)
                box_center.append(bbox_convert_center(box2d))
                box2d = box2d.astype(np.int32)
                box_pos.append(box2d)


                p1, p2 = box2d[:2], box2d[2:]
                box_mask[p1[1]:p2[1], p1[0]:p2[0]] = 1
                box_mask[0,0] = box_label
                
                # print(f'box name: {box.name}, num points in box: {sum(cur_mask)}')
                bool_mask = cur_mask.astype(bool)
                box_radar_align[bool_mask] = box_count

                radar_proj_height[cur_mask] = box.wlh[2]

                mask = np.logical_or(mask, cur_mask)
                box_mask_all[..., box_count-1] = box_mask

                box_count += 1
    box_pos_return = copy.deepcopy(box_pos)

    order = np.argsort(box_dist)[::-1]
    box_pos = [box_pos[i] for i in order]
    box_height = [box_height[i] for i in order]
    for idx, box in enumerate(box_pos):
        box_height_map[box[1]:box[3], box[0]:box[2]] = box_height[idx]

    mask = np.clip(mask, a_min=0, a_max=1)
    # if box_mask_all:
    #     box_mask_all = np.stack(box_mask_all, axis=-1)
    if box_count>1:
        box_center = np.stack(box_center, axis=0) 
    
    # add randomness
    # radar_proj_height[radar_proj_height==0] = np.random.uniform(0, 0.2, size=radar_proj_height[radar_proj_height==0].shape)

    return mask, box_radar_align, box_mask_all, box_count-1, box_center, box_height_map, box_pos_return

def merge_radar_point_clouds(nusc,
                             nusc_explorer,
                             current_sample_token,
                             n_forward,
                             n_backward):
    '''
    Merges Radar point from multiple samples and adds them to a single depth image
    Picks current_sample_token as reference and projects lidar points from all other frames into current_sample.

    Arg(s):
        nusc : NuScenes Object
            nuScenes object instance
        nusc_explorer : NuScenesExplorer Object
            nuScenes explorer object instance
        current_sample_token : str
            token for accessing the current sample data
        n_forward : int
            number of frames to merge in the forward direction.
        n_backward : int
            number of frames to merge in the backward direction
    Returns:
        numpy[float32] : 2 x N of x, y for radar points projected into the image
        numpy[float32] : N depths of radar points

    '''
    # category selection
    cartesian_uncertainty = 0.5 # meters
    angular_uncertainty = math.radians(1.7) # degree
    class_weight = {
        'vehicle.car' : 1,
        'vehicle.motorcycle' : 1,
        'vehicle.bicycle' : 1,
        'vehicle.bus.rigid' : 1,
        'vehicle.truck' : 1,
        'vehicle.trailer' : 1,
        'human.pedestrian.adult' : 1
                        }

    category_selection = class_weight.keys()
    category_mapping = {
                        "vehicle.car" : "vehicle.car",
                        "vehicle.motorcycle" : "vehicle.motorcycle",
                        "vehicle.bicycle" : "vehicle.bicycle",
                        "vehicle.bus" : "vehicle.bus",
                        "vehicle.truck" : "vehicle.truck",
                        "vehicle.emergency" : "vehicle.truck",
                        "vehicle.trailer" : "vehicle.trailer",
                        "human" : "human", }

    classes, labels = _get_class_label_mapping([c['name'] for c in nusc.category], category_mapping)

    # Get the sample
    current_sample = nusc.get('sample', current_sample_token)

    # Get lidar token in the current sample
    main_radar_token = current_sample['data']['RADAR_FRONT']

    # Get the camera token for the current sample
    main_camera_token = current_sample['data']['CAM_FRONT']

    rad = nusc.get_sample_data(main_radar_token)
    cam = nusc.get_sample_data(main_camera_token)
    nusc_sample_data_rad = nusc.get('sample_data', main_radar_token)
    nusc_sample_data_cam = nusc.get('sample_data', main_camera_token)

    # load radar pc
    RadarPointCloud.disable_filters()
    pc = RadarPointCloud.from_file(rad[0])

    # load image
    image = Image.open(cam[0])
    image = np.array(image)
    image_shape = image.shape

    # find radar box alignment
    radar_gt_mask, box_radar_align, box_mask_all, box_count, box_center, box_height_map, box_pos = calc_mask_mod(
                                     nusc=nusc, nusc_sample_data_rad=nusc_sample_data_rad, \
                                     nusc_sample_data_cam=nusc_sample_data_cam, points3d=pc.points[0:3,:], \
                                     image_min_side = image_shape[0], image_max_side = image_shape[1],\
                                     tolerance=cartesian_uncertainty, angle_tolerance=angular_uncertainty, \
                                     category_selection=category_selection, classes=classes)

    if len(box_pos)!=0:
        box_pos = np.stack(box_pos, axis=0)
        assert box_pos.shape[0] == box_count, 'box number and pos might be wrong!'
    
    radar_points = np.concatenate([pc.points, box_radar_align[np.newaxis, :]], axis=0)
    radar_pc = copy.deepcopy(radar_points)
    pc_copy = copy.deepcopy(pc)

    main_points_radar, main_depth_radar, main_image = point_cloud_to_image(nusc,
                         radar_pc,
                         main_radar_token,
                         main_camera_token,
                         min_distance_from_camera=1.0,
                         include_feature=True)

    main_feature_radar = np.concatenate([main_depth_radar[np.newaxis, :], main_points_radar[5:8, :], main_points_radar[-1:, :]], axis=0)


    # Create an empty radar image
    main_radar_image = np.zeros((image_shape[0], image_shape[1], main_feature_radar.shape[0]))

    main_points_radar_quantized = np.round(main_points_radar[:2, :]).astype(int)

    # Iterating through each radar point and plotting them onto the radar image
    for point_idx in range(0, main_points_radar_quantized.shape[1]):
        # Get x and y index in image frame
        x = main_points_radar_quantized[0, point_idx]
        y = main_points_radar_quantized[1, point_idx]

        # Value of y, x is the depth
        main_radar_image[y, x, :] = main_feature_radar[:, point_idx]

    # Create a validity map to check which elements of the radar image are valid
    main_validity_map = np.where(main_radar_image[..., 0] > 0, 1, 0)

    # Count forward and backward frames
    n_forward_processed = 0
    n_backward_processed = 0

    # Initialize next sample as current sample
    next_sample = copy.deepcopy(current_sample)

    while next_sample['next'] != "" and n_forward_processed < n_forward:

        '''
        1. Load point cloud in `next' frame,
        2. Project onto image to remove vehicle bounding boxes
        3. Backproject to camera frame
        '''

        # Get the token and sample data for the next sample amd move forward
        next_sample_token = next_sample['next']
        next_sample = nusc.get('sample', next_sample_token)

        # Get radar and camera token in the current sample
        next_radar_token = next_sample['data']['RADAR_FRONT']
        next_camera_token = next_sample['data']['CAM_FRONT']
        
        next_rad = nusc.get_sample_data(next_radar_token)
        next_cam = nusc.get_sample_data(next_camera_token)

        # Grab the radar camera sample
        next_radar_sample = nusc.get('sample_data', next_radar_token)
        next_camera_sample = nusc.get('sample_data', next_camera_token)

        # get the point cloud path and grab the radar point cloud
        RadarPointCloud.disable_filters()
        next_pc = RadarPointCloud.from_file(next_rad[0])
        
        # calculate radar box alignment
        next_radar_gt_mask, next_box_radar_align, next_box_mask_all, next_box_count, next_box_center, next_box_height_map,\
            next_box_pos = calc_mask_mod(
                                        nusc=nusc, nusc_sample_data_rad=nusc_sample_data_rad, \
                                        nusc_sample_data_cam=nusc_sample_data_cam, points3d=next_pc.points[0:3,:], \
                                        image_min_side = image_shape[0], image_max_side = image_shape[1],\
                                        tolerance=cartesian_uncertainty, angle_tolerance=angular_uncertainty, \
                                        category_selection=category_selection, classes=classes)

        next_radar_points = np.concatenate([next_pc.points, next_box_radar_align[np.newaxis, :]], axis=0)
        next_radar_pc = copy.deepcopy(next_radar_points)
        next_pc_copy = copy.deepcopy(next_pc)

        # Project the points to the image frame of reference as 2 x N x, y and 1 x N z arrays
        next_points_radar_main, next_depth_radar_main, _ = point_cloud_to_image(
            nusc=nusc,
            point_cloud=next_radar_pc,
            lidar_sensor_token=next_radar_token,
            camera_token=main_camera_token,
            min_distance_from_camera=1.0, 
            include_feature=True)
        
        next_feature_radar = np.concatenate([next_depth_radar_main[np.newaxis, :],\
                                            next_points_radar_main[5:8, :], next_points_radar_main[-1:, :]], axis=0)
        next_points_radar_main_quantized = np.round(next_points_radar_main[:2, :]).astype(int)

        for point_idx in range(0, next_points_radar_main_quantized.shape[1]):
            x = next_points_radar_main_quantized[0, point_idx]
            y = next_points_radar_main_quantized[1, point_idx]

            is_not_occluded = \
                main_validity_map[y, x] == 1 and \
                next_depth_radar_main[point_idx] < main_radar_image[y, x, 0]

            if is_not_occluded:
                main_radar_image[y, x, :] = next_feature_radar[:, point_idx]
            elif main_validity_map[y, x] != 1:
                main_radar_image[y, x, :] = next_feature_radar[:, point_idx]
                main_validity_map[y, x] = 1

        n_forward_processed = n_forward_processed + 1

    # Initialize previous sample as current sample
    prev_sample = copy.deepcopy(current_sample)

    while prev_sample['prev'] != "" and n_backward_processed < n_backward:
        '''
        1. Load point cloud in `prev' frame,
        2. Poject onto image to remove vehicle bounding boxes
        3. Backproject to camera frame
        '''

        # Get the token and sample data for the prev sample and move forward
        prev_sample_token = prev_sample['prev']
        prev_sample = nusc.get('sample', prev_sample_token)

        # Get radar and camera token in the current sample
        prev_radar_token = prev_sample['data']['RADAR_FRONT']
        prev_rad = nusc.get_sample_data(prev_radar_token)


        # Grab the radar sample
        prev_radar_sample = nusc.get('sample_data', prev_radar_token)

        # get the point cloud path and grab the radar point cloud
        RadarPointCloud.disable_filters()
        prev_pc = RadarPointCloud.from_file(prev_rad[0])
        
        # calculate radar box alignment
        prev_radar_gt_mask, prev_box_radar_align, prev_box_mask_all, prev_box_count, prev_box_center, prev_box_height_map,\
            prev_box_pos = calc_mask_mod(
                                        nusc=nusc, nusc_sample_data_rad=nusc_sample_data_rad, \
                                        nusc_sample_data_cam=nusc_sample_data_cam, points3d=prev_pc.points[0:3,:], \
                                        image_min_side = image_shape[0], image_max_side = image_shape[1],\
                                        tolerance=cartesian_uncertainty, angle_tolerance=angular_uncertainty, \
                                        category_selection=category_selection, classes=classes)
        
        prev_radar_points = np.concatenate([prev_pc.points, prev_box_radar_align[np.newaxis, :]], axis=0)
        prev_radar_pc = copy.deepcopy(prev_radar_points)
        prev_pc_copy = copy.deepcopy(prev_pc)


        # Project the points to the image frame of reference as 2 x N x, y and 1 x N z arrays
        prev_points_radar_main, prev_depth_radar_main, _ = point_cloud_to_image(
            nusc=nusc,
            point_cloud=prev_radar_pc,
            lidar_sensor_token=prev_radar_token,
            camera_token=main_camera_token,
            min_distance_from_camera=1.0,
            include_feature=True)

        prev_feature_radar = np.concatenate([prev_depth_radar_main[np.newaxis, :],\
                                                prev_points_radar_main[5:8, :], prev_points_radar_main[-1:, :]], axis=0)
        prev_points_radar_main_quantized = np.round(prev_points_radar_main[:2, :]).astype(int)

        for point_idx in range(0, prev_points_radar_main_quantized.shape[1]):
            x = prev_points_radar_main_quantized[0, point_idx]
            y = prev_points_radar_main_quantized[1, point_idx]

            is_not_occluded = \
                main_validity_map[y, x] == 1 and \
                prev_depth_radar_main[point_idx] < main_radar_image[y, x, 0]

            if is_not_occluded:
                main_radar_image[y, x, :] = prev_feature_radar[:, point_idx]
            elif main_validity_map[y, x] != 1:
                main_radar_image[y, x, :] = prev_feature_radar[:, point_idx]
                main_validity_map[y, x] = 1

        n_backward_processed = n_backward_processed + 1

    # need to convert this to the same format used by nuScenes to return Lidar points
    # nuscenes outputs this in the form of a xy tuple and depth. We do the same here.
    # we also make x -> y and y -> x to stay consistent with nuScenes
    return_points_radar_y, return_points_radar_x = np.nonzero(main_radar_image[..., 0])

    # Array of 5, N features: depth, rcs, vx, vy, box_radar_align
    return_feature_radar = main_radar_image[return_points_radar_y, return_points_radar_x, :]

    # Array of 2, N x, y coordinates for lidar, swap (y, x) components to (x, y)
    return_points_radar = np.stack([
        return_points_radar_x,
        return_points_radar_y],
        axis=-1)

    return_points_radar = np.concatenate([return_points_radar, return_feature_radar], axis=-1)

    return return_points_radar, box_pos

def lidar_depth_map_from_token(nusc,
                               nusc_explorer,
                               current_sample_token):
    '''
    Picks current_sample_token as reference and projects lidar points onto the image plane.

    Arg(s):
        nusc : NuScenes Object
            nuScenes object instance
        nusc_explorer : NuScenesExplorer Object
            nuScenes explorer object instance
        current_sample_token : str
            token for accessing the current sample data
    Returns:
        numpy[float32] : H x W depth
    '''

    current_sample = nusc.get('sample', current_sample_token)
    lidar_token = current_sample['data']['LIDAR_TOP']
    main_camera_token = current_sample['data']['CAM_FRONT']

    # project the lidar frame into the camera frame
    main_points_lidar, main_depth_lidar, main_image = nusc_explorer.map_pointcloud_to_image(
        pointsensor_token=lidar_token,
        camera_token=main_camera_token)

    depth_map = points_to_depth_map(main_points_lidar, main_depth_lidar, main_image)

    return depth_map

def points_to_depth_map(points, depth, image):
    '''
    Plots the depth values onto the image plane

    Arg(s):
        points : numpy[float32]
            2 x N matrix in x, y
        depth : numpy[float32]
            N scales for z
        image : numpy[float32]
            H x W x 3 image for reference frame size
    Returns:
        numpy[float32] : H x W image with depth plotted
    '''

    # Plot points onto the image
    image = np.asarray(image)
    depth_map = np.zeros((image.shape[0], image.shape[1]))

    points_quantized = np.round(points).astype(int)

    for pt_idx in range(0, points_quantized.shape[1]):
        x = points_quantized[0, pt_idx]
        y = points_quantized[1, pt_idx]
        depth_map[y, x] = depth[pt_idx]

    return depth_map

def process_scene(args):
    '''
    Processes one scene from first sample to last sample

    Arg(s):
        args : tuple(Object, Object, str, int, str, str, int, int, str, bool)
            nusc : NuScenes Object
                nuScenes object instance
            nusc_explorer : NuScenesExplorer Object
                nuScenes explorer object instance
            tag : str
                train, val
            scene_id : int
                identifier for one scene
            panoptic_seg_dir : str
                directory where all the panoptic segmentation masks are stored
            first_sample_token : str
                token to identify first sample in the scene for fetching
            last_sample_token : str
                token to identify last sample in the scene for fetching
            n_forward : int
                number of forward (future) frames to reproject
            n_backward : int
                number of backward (previous) frames to reproject
            output_dirpath : str
                root of output directory
            paths_only : bool
                if set, then only produce paths
    Returns:
        list[str] : paths to camera image
        list[str] : paths to lidar depth map
        list[str] : paths to radar depth map
        list[str] : paths to ground truth (merged lidar) depth map
        list[str] : paths to ground truth (merged lidar) interpolated depth map
    '''

    tag, \
        scene_id, \
        first_sample_token, \
        last_sample_token, \
        n_forward, \
        n_backward, \
        output_dirpath, \
        paths_only = args

    # Instantiate the first sample id
    sample_id = 0
    sample_token = first_sample_token

    radar_points_paths = []
    radar_points_reprojected_paths = []
    box_pos_paths = []
    
    print('Processing scene_id={}'.format(scene_id))

    # Iterate through all samples up to the last sample
    while sample_token != last_sample_token:

        # Fetch a single sample
        current_sample = nusc.get('sample', sample_token)
        camera_token = current_sample['data']['CAM_FRONT']
        camera_sample = nusc.get('sample_data', camera_token)

        '''
        Set up paths
        '''
        camera_image_path = os.path.join(nusc.dataroot, camera_sample['filename'])

        dirpath, filename = os.path.split(camera_image_path)
        dirpath = dirpath.replace(nusc.dataroot, output_dirpath)
        filename = os.path.splitext(filename)[0]


        # Create radar path
        radar_points_dirpath = dirpath.replace(
            'samples',
            os.path.join('radar_points_new', 'scene_{}'.format(scene_id)))
        radar_points_filename = filename + '.npy'

        radar_points_path = os.path.join(
            radar_points_dirpath,
            radar_points_filename)

        radar_points_reprojected_dirpath = dirpath.replace(
            'samples',
            os.path.join('radar_points_reprojected_new', 'scene_{}'.format(scene_id)))

        radar_points_reprojected_path = os.path.join(
            radar_points_reprojected_dirpath,
            radar_points_filename)

        # Create box mask path
        box_pos_dirpath = dirpath.replace(
            'samples',
            os.path.join('box_pos', 'scene_{}'.format(scene_id)))
        box_pos_filename = filename + '.npy'

        box_pos_path = os.path.join(
            box_pos_dirpath,
            box_pos_filename)


        # In case multiple threads create same directory
        dirpaths = [
            radar_points_dirpath,
            radar_points_reprojected_dirpath,
            box_pos_dirpath,

        ]

        for dirpath in dirpaths:
            if not os.path.exists(dirpath):
                try:
                    os.makedirs(dirpath)
                except Exception:
                    pass

        '''
        Store file paths
        '''
        radar_points_paths.append(radar_points_path)
        radar_points_reprojected_paths.append(radar_points_reprojected_path)
        box_pos_paths.append(box_pos_path)

        if not paths_only:

            '''
            Get camera data
            '''
            camera_image = data_utils.load_image(camera_image_path)

            '''
            Merge forward and backward point clouds for radar and lidar
            '''
            # Transform Lidar and Radar Points to the image coordinate
            return_points_radar_reprojected, box_pos_all_reprojected = merge_radar_point_clouds(
                nusc=nusc,
                nusc_explorer=nusc_explorer,
                current_sample_token=sample_token,
                n_forward=n_forward,
                n_backward=n_backward)

            return_points_radar, box_pos = merge_radar_point_clouds(
                nusc=nusc,
                nusc_explorer=nusc_explorer,
                current_sample_token=sample_token,
                n_forward=0,
                n_backward=0)


            '''
            Save radar points as a numpy array
            '''

            np.save(radar_points_reprojected_path, return_points_radar_reprojected)
            np.save(radar_points_path, return_points_radar)

            '''
            Store box mask
            '''
            np.save(box_pos_path, box_pos)

        '''
        Move to next sample in scene
        '''
        sample_id = sample_id + 1
        sample_token = current_sample['next']

    print('Finished {} samples in scene_id={}'.format(sample_id, scene_id))

    return (tag,
            radar_points_paths,
            radar_points_reprojected_paths,
            box_pos_paths
            )


'''
Main function
'''
if __name__ == '__main__':

    use_multithread = args.n_thread > 1 and not args.debug

    pool_inputs = []
    pool_results = []

    test_radar_points_new_paths = []
    test_radar_points_reprojected_new_paths = []
    test_box_pos_paths = []

    n_scenes_to_process = min(args.n_scenes_to_process, MAX_SCENES)
    print('Total Scenes to process: {}'.format(n_scenes_to_process))

    # Add all tasks for processing each scene to pool inputs
    for scene_id in range(0, min(args.n_scenes_to_process, MAX_SCENES)):
    # for scene_id in [710]:

        tag = 'test'
        current_scene = nusc.scene[scene_id]
        first_sample_token = current_scene['first_sample_token']
        last_sample_token = current_scene['last_sample_token']

        inputs = [
            tag,
            scene_id,
            first_sample_token,
            last_sample_token,
            args.n_forward_frames_to_reproject,
            args.n_backward_frames_to_reproject,
            args.nuscenes_data_derived_dirpath,
            args.paths_only
        ]

        pool_inputs.append(inputs)

        if not use_multithread:
            pool_results.append(process_scene(inputs))

    if use_multithread:
        # Create pool of threads
        with mp.Pool(args.n_thread) as pool:
            # Will fork n_thread to process scene
            pool_results = pool.map(process_scene, pool_inputs)

    # Unpack output paths
    for results in pool_results:

        tag, \
            radar_points_scene_paths,\
            radar_points_reprojected_scene_paths,\
            box_pos_scene_paths = results

        test_radar_points_new_paths.extend(radar_points_scene_paths)
        test_radar_points_reprojected_new_paths.extend(radar_points_reprojected_scene_paths)
        test_box_pos_paths.extend(box_pos_scene_paths)


    '''
    Write paths to file
    '''
    outputs = [
        [
            'training',
            [
                [
                    'radar new',
                    test_radar_points_new_paths,
                    TEST_RADAR_NEW_FILEPATH
                ], [
                    'radar reprojected new',
                    test_radar_points_reprojected_new_paths,
                    TEST_RADAR_REPROJECTED_NEW_FILEPATH,
                ], [
                    'box mask',
                    test_box_pos_paths,
                    TEST_BOX_POS_FILEPATH,
                ],
            ]
        ]
    ]

    # Create output directories
    for dirpath in [TEST_REF_DIRPATH]:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    for output_info in outputs:

        tag, output = output_info
        for output_type, paths, filepath in output:

            print('Storing {} {} {} file paths into: {}'.format(
                len(paths), tag, output_type, filepath))
            data_utils.write_paths(filepath, paths)
