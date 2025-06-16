from typing import Iterator, Tuple, Any
import traceback
import numpy as np
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import json
from PIL import Image

import random
from skimage.measure import block_reduce
from scipy.spatial.transform import Rotation
from droid.droid_utils import load_trajectory, crawler
from droid.tfds_utils import MultiThreadedDatasetBuilder
import warnings

# Modify to point to directory with raw DROID MP4 data
DATA_PATH = "/vault/CHORDSkills/DROID3D_RAW"
VER = "1.2.0"  # version of the dataset
# Find the file called aggregated-annotations in DATA_PATH
ANNOTATION_PATH = None
for fname in os.listdir(DATA_PATH):
    if fname.startswith("aggregated-annotations"):
        ANNOTATION_PATH = os.path.join(DATA_PATH, fname)
        break
if ANNOTATION_PATH is None:
    print("aggregated-annotations file not found in DATA_PATH.")
else:
    print(f"Found aggregated-annotations file: {ANNOTATION_PATH}")
    
with open(ANNOTATION_PATH, 'r') as f:
    annotations = json.load(f)
    
IMAGE_RES = (180, 320)
MAX_DEPTH = 3  # maximum depth value in meters

use_depth = True

def get_cam_extrinsics(metadata, data, i):
    _data = data[i]  # assuming all data has the same structure
    CAMERA_NAMES = ["ext1", "ext2", "wrist"]
    serial = {
        camera_name: metadata[f"{camera_name}_cam_serial"]
        for camera_name in CAMERA_NAMES
    }
    extrinsics = {}
    for camera_name, serial_key in serial.items():
        # Dummy values for extrinsics if not available
        extrinsics[camera_name + "_left"] = np.zeros(6, dtype=np.float32)
        extrinsics[camera_name + "_right"] = np.zeros(6, dtype=np.float32)     
        if f"{serial_key}_left" in _data["observation"]["camera_extrinsics"]:
            extrinsics[camera_name + "_left"] = np.array(_data["observation"]["camera_extrinsics"][
                f"{serial_key}_left"
            ]).astype(np.float32)
            
        if f"{serial_key}_right" in _data["observation"]["camera_extrinsics"]:
            extrinsics[camera_name + "_right"] = np.array(_data["observation"]["camera_extrinsics"][
                f"{serial_key}_right"
            ]).astype(np.float32)
            
    return extrinsics

def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    def _resize_and_encode(image, size):
            # Resize
            resized = Image.fromarray(image).resize(size, resample=Image.BICUBIC)
            resized_np = np.array(resized)
            return resized_np
        
    def downsample_depth_median(depth, k=4):
        # If all depth values are zero, return a zero array
        if np.all(depth == 0):
            return np.zeros((IMAGE_RES[0], IMAGE_RES[1]), dtype=np.float32)
        # Replace 0 with nan so they don’t pollute the median
        depth_np = np.where(depth == 0, np.nan, depth)
        with np.errstate(invalid="ignore"):
            reduced_depth = block_reduce(depth_np, block_size=(k, k), func=np.nanmedian)
        reduced_depth = np.nan_to_num(reduced_depth, nan=0.0)  # Replace nans with 0
        H, W = reduced_depth.shape
        if H != IMAGE_RES[0] or W != IMAGE_RES[1]:
            depth_img = Image.fromarray(reduced_depth, mode="F")
            depth_img_resized = depth_img.resize((IMAGE_RES[1], IMAGE_RES[0]), resample=Image.NEAREST)
            reduced_depth =  np.array(depth_img_resized)
        H, W = reduced_depth.shape
        assert H == IMAGE_RES[0] and W == IMAGE_RES[1], f"Unexpected depth resolution: {H}x{W} != {IMAGE_RES[0]}x{IMAGE_RES[1]}"
        return reduced_depth

    def _parse_example(episode_path):
        h5_filepath = os.path.join(episode_path, 'trajectory.h5')
        recording_folderpath = os.path.join(episode_path, 'recordings', 'MP4')
        print(f"Processing episode: {episode_path}")
        if use_depth:
            import pyzed.sl as sl
        try:
            data = load_trajectory(h5_filepath, recording_folderpath=recording_folderpath)
        except:
           print(f"Skipping trajectory because data couldn't be loaded for {episode_path}.")
           return None

        # get language instruction -- modify if more than one instruction
        # Find the metadata JSON file (named meta_<HASH>.json) in the episode path
        json_files = [f for f in os.listdir(episode_path) if f.startswith('metadata_') and f.endswith('.json')]
        if not json_files:
            print(f"Skipping trajectory because metadata JSON not found for {episode_path}.")
            return None
        json_path = os.path.join(episode_path, json_files[0])
        with open(json_path, 'r') as f:
            metadata = json.load(f)
            
        TASKS_ID = metadata.get('uuid', '')
        if TASKS_ID not in annotations:
            print(f"Skipping trajectory because no annotations found for {TASKS_ID} in {ANNOTATION_PATH}.")
            return None
        lang = ''
        lang_entry = annotations.get(TASKS_ID, {})
        if isinstance(lang_entry, dict):
            lang_keys = [k for k in lang_entry.keys() if k.startswith('language_instruction')]
            if lang_keys:
                chosen_key = random.choice(lang_keys)
                lang = lang_entry[chosen_key]
                
        # if use_depth is True, initialize a ZED camera for each view
        if use_depth:
            depth_cams = {}
            for cam_name in ["ext1", "ext2", "wrist"]:
                serial_val = metadata.get(f"{cam_name}_cam_serial")
                init_params = sl.InitParameters()
                init_params.sdk_verbose = 0  
                svo_path = os.path.join(episode_path, "recordings", "SVO", f"{serial_val}.svo")
                init_params.set_from_svo_file(svo_path)
                init_params.depth_mode = sl.DEPTH_MODE.QUALITY
                init_params.svo_real_time_mode = False
                init_params.coordinate_units = sl.UNIT.METER
                init_params.depth_minimum_distance = 0.2
                cam = sl.Camera()
                err = cam.open(init_params)
                if err != sl.ERROR_CODE.SUCCESS:
                    print(f"Error reading camera data for {cam_name}: {err}")
                    depth_cams[cam_name] = None
                else:
                    depth_cams[cam_name] = cam
        
        try:
            assert all(t.keys() == data[0].keys() for t in data)

            episode = []
            intrinsics = {}
            cam_extrinsics = get_cam_extrinsics(metadata, data, 0)
            for i, step in enumerate(data):
                obs = step['observation']
                action = step['action']
                camera_type_dict = obs['camera_type']
                wrist_ids = [k for k, v in camera_type_dict.items() if v == 0]
                exterior_ids = [k for k, v in camera_type_dict.items() if v != 0]
                _rgb_map = {
                    'ext1': f'{exterior_ids[0]}_left',
                    'ext2': f'{exterior_ids[1]}_left',
                    'wrist': f'{wrist_ids[0]}_left'
                }
                
                # if depth is desired, grab a new depth image from each camera
                if use_depth:
                    depth_images = {}
                    for cam_name, cam in depth_cams.items():
                        if cam is not None:
                            rt_param = sl.RuntimeParameters()
                            err = cam.grab(rt_param)
                            if err == sl.ERROR_CODE.SUCCESS:
                                depth_mat = sl.Mat()
                                cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
                                # Convert to numpy array. Depending on the version of the API you can use:
                                depth_np = np.array(depth_mat.get_data())
                                depth_images[cam_name] = depth_np
                            elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                                depth_images[cam_name] = np.zeros((*IMAGE_RES,), dtype=np.float32)

                        else:
                            depth_images[cam_name] = np.zeros((*IMAGE_RES,), dtype=np.float32)
                
                for cam_name, cam in depth_cams.items():
                    rgb = obs['image'][_rgb_map[cam_name]].copy().astype(np.uint8)
                    HD_RES = rgb.shape[:2]
                    if cam is not None:
                        # Read RGB and depth images
                        depth = np.copy(depth_images[cam_name]).astype(np.float32)
                        # Make sure rgb and depth images have the same resolution
                        if np.all(depth == 0): # Handle empty frames
                            depth = np.zeros((*HD_RES,), dtype=np.float32)
                        assert rgb.shape[:2] == HD_RES, f"RGB and depth images have different resolutions: {rgb.shape[:2]} vs {HD_RES}"
                        # Resize images to downsampled resolution
                        rgb = _resize_and_encode(rgb, (IMAGE_RES[1], IMAGE_RES[0]))
                        # Downsample depth map using median filter
                        depth = downsample_depth_median(depth, k = HD_RES[0] // IMAGE_RES[0])
                        depth = np.expand_dims(depth, axis=-1)
                        depth[depth > MAX_DEPTH] = 0  # filter out invalid depth values
                        # Read camera intrinsics for the first frame only
                        if i == 0:
                            params = (cam.get_camera_information().camera_configuration.calibration_parameters)
                            
                            # Adjust intrinsic parameters based on the scale factor
                            left_intrinsic_mat = np.array(
                                [
                                    [params.left_cam.fx , 0, params.left_cam.cx],
                                    [0, params.left_cam.fy, params.left_cam.cy],
                                    [0, 0, 1],
                                ]
                            , dtype=np.float32)
                            intrinsic  = left_intrinsic_mat.copy()
                            # Adjust the intrinsic matrix for the resolution change
                            s_h = HD_RES[0] / IMAGE_RES[0]
                            s_w = HD_RES[1] / IMAGE_RES[1]
                            intrinsic[0, 0] /= s_w   # fx'
                            intrinsic[1, 1] /= s_h   # fy'
                            intrinsic[0, 2] /= s_w   # cx'
                            intrinsic[1, 2] /= s_h   # cy
                            intrinsics[cam_name] = intrinsic # Store intrinsics only once per episode
                        # Update RGB and depth images
                        obs['image'][_rgb_map[cam_name]] = rgb
                        depth_images[cam_name] = depth
                        # # Build point cloud from depth map
                        # H, W, _ = depth.shape
                        # fx, fy = intrinsic[0, 0], intrinsic[1, 1]
                        # cx, cy = intrinsic[0, 2], intrinsic[1, 2]
                    
                        # i_grid, j_grid = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
                        # z = depth.flatten()
                        # x = ((i_grid.flatten() - cx) * z) / fx
                        # y = ((j_grid.flatten() - cy) * z) / fy

                        # points = np.stack((x, y, z), axis=-1)
                        # valid = (z > 0) & np.isfinite(z)
                        # points = points[valid]
                        # if not np.any(valid):
                        #     colors = np.zeros((0, 3), dtype=np.uint8)
                        # else:
                        #     colors = rgb.reshape(-1, 3)[valid]                        
                        #     # Apply extrinsics to the points
                        #     extrinsics = np.array(extrinsics, dtype=np.float32)
                        #     rotation = Rotation.from_euler("xyz", np.array(extrinsics[3:])).as_matrix().astype(np.float32)
                        #     points = (rotation @ points.T).T + extrinsics[:3]
                        # n_points = IMAGE_RES[0] * IMAGE_RES[1]
                        # # Pad points and colors to have the same length
                        # if points.shape[0] < n_points:
                        #     padding = np.zeros((n_points - points.shape[0], 3), dtype=np.float32)
                        #     points = np.vstack((points, padding))
                        #     colors = np.vstack((colors, np.zeros((n_points - colors.shape[0], 3), dtype=np.uint8)))
                        # elif points.shape[0] > n_points:
                        #     points = points[:n_points]
                        #     colors = colors[:n_points]
                       
                        # # Store point cloud
                        # pcd[cam_name] = {
                        #     'points': points.astype(np.float32),
                        #     'colors': colors.astype(np.uint8)
                        # }
                # Check data
                
                episode.append({
                    'observation': {
                        # RGB images
                        'exterior_image_1_left': obs['image'][f'{exterior_ids[0]}_left'][..., ::-1],
                        'exterior_image_2_left': obs['image'][f'{exterior_ids[1]}_left'][..., ::-1],
                        'wrist_image_left': obs['image'][f'{wrist_ids[0]}_left'][..., ::-1],
                        # Include depth information if available; otherwise, these keys can be ignored downstream.
                        'exterior_depth_1_left': depth_images["ext1"] if use_depth else np.zeros((*IMAGE_RES, 1), dtype=np.float32),
                        'exterior_depth_2_left': depth_images["ext2"] if use_depth else np.zeros((*IMAGE_RES, 1), dtype=np.float32),
                        'wrist_depth_left': depth_images["wrist"] if use_depth else np.zeros((*IMAGE_RES, 1), dtype=np.float32),
                        # Point clouds
                        # 'exterior_pc_1_left': pcd['ext1'],
                        # 'exterior_pc_2_left': pcd['ext2'],
                        # 'wrist_pc_left': pcd['wrist'],
                        # Robot state information
                        'cartesian_position': obs['robot_state']['cartesian_position'],
                        'joint_position': obs['robot_state']['joint_positions'],
                        'gripper_position': np.array([obs['robot_state']['gripper_position']]),
                    },
                    'action_dict': {
                        'cartesian_position': action['cartesian_position'],
                        'cartesian_velocity': action['cartesian_velocity'],
                        'gripper_position': np.array([action['gripper_position']]),
                        'gripper_velocity': np.array([action['gripper_velocity']]),
                        'joint_position': action['joint_position'],
                        'joint_velocity': action['joint_velocity'],
                    },
                    'action': np.concatenate((action['cartesian_position'], [action['gripper_position']])),
                    'discount': 1.0,
                    'reward': float((i == (len(data) - 1) and 'success' in episode_path)),
                    'is_first': i == 0,
                    'is_last': i == (len(data) - 1),
                    'is_terminal': i == (len(data) - 1)
                })
            # close depth cameras
            if use_depth:
                for cam in depth_cams.values():
                    if cam is not None:
                        cam.close()
        except:
           print(f"Skipping trajectory because there was an error in data processing for {episode_path}.")
           # Print traceback for debugging
           traceback.print_exc()  
           if use_depth:
               for cam in depth_cams.values():
                   if cam is not None:
                       cam.close()
           return None

        # # create output data sample
        sample = {
            'language_instruction': lang,
            'steps': episode,
            'episode_metadata': {
                'file_path': h5_filepath,
                'recording_folderpath': recording_folderpath,
                'cam_extrinsics': cam_extrinsics,
                'cam_intrinsics': intrinsics,
            },
        }
        # # if you want to skip an example for whatever reason, simply return None
        return episode_path, sample

    # for smallish datasets, use single-thread parsing
    for sample in paths:
        yield _parse_example(sample)
        # _parse_example(sample)


class Droid(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version(VER)
    RELEASE_NOTES = {
      VER: 'Fixed RGB parsing',
    }

    N_WORKERS = 6                  # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 3       # number of paths converted & stored in memory before writing to disk
                                    # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                                    # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples  # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'language_instruction': tfds.features.Text(
                    doc='Language Instruction.'
                ),
                'steps': tfds.features.Dataset({
                        'observation': tfds.features.FeaturesDict({
                            'exterior_image_1_left': tfds.features.Image(
                                shape=(*IMAGE_RES, 3),
                                dtype=np.uint8,
                                encoding_format='jpeg',
                                doc='Exterior camera 1 left viewpoint',
                            ),
                            'exterior_image_2_left': tfds.features.Image(
                                shape=(*IMAGE_RES, 3),
                                dtype=np.uint8,
                                encoding_format='jpeg',
                                doc='Exterior camera 2 left viewpoint'
                            ),
                            'wrist_image_left': tfds.features.Image(
                                shape=(*IMAGE_RES, 3),
                                dtype=np.uint8,
                                encoding_format='jpeg',
                                doc='Wrist camera RGB left viewpoint',
                            ),
                            'exterior_depth_1_left': tfds.features.Tensor(
                                shape=(*IMAGE_RES, 1),
                                dtype=np.float32,
                                doc='Depth map for exterior camera 1 left viewpoint.'
                            ),
                            'exterior_depth_2_left': tfds.features.Tensor(
                                shape=(*IMAGE_RES, 1),
                                dtype=np.float32,
                                doc='Depth map for exterior camera 2 left viewpoint.'
                            ),
                            'wrist_depth_left': tfds.features.Tensor(
                                shape=(*IMAGE_RES, 1),
                                dtype=np.float32,
                                doc='Depth map for wrist camera left viewpoint.'
                            ),
                            # 'exterior_pc_1_left': tfds.features.FeaturesDict({
                            #     'points': tfds.features.Tensor(
                            #         shape=((IMAGE_RES[0]*IMAGE_RES[1]), 3),
                            #         dtype=np.float32,
                            #         doc='Point cloud for exterior camera 1 left viewpoint.'
                            #     ),
                            #     'colors': tfds.features.Tensor(
                            #         shape=((IMAGE_RES[0]*IMAGE_RES[1]), 3),
                            #         dtype=np.uint8,
                            #         doc='Colors for point cloud of exterior camera 1 left viewpoint.'
                            #     )
                            # }),
                            # 'exterior_pc_2_left': tfds.features.FeaturesDict({
                            #     'points': tfds.features.Tensor(
                            #         shape=((IMAGE_RES[0]*IMAGE_RES[1]), 3),
                            #         dtype=np.float32,
                            #         doc='Point cloud for exterior camera 2 left viewpoint.'
                            #     ),
                            #     'colors': tfds.features.Tensor(
                            #         shape=((IMAGE_RES[0]*IMAGE_RES[1]), 3),
                            #         dtype=np.uint8,
                            #         doc='Colors for point cloud of exterior camera 2 left viewpoint.'
                            #     )
                            # }),
                            # 'wrist_pc_left': tfds.features.FeaturesDict({
                            #     'points': tfds.features.Tensor(
                            #         shape=((IMAGE_RES[0]*IMAGE_RES[1]), 3),
                            #         dtype=np.float32,
                            #         doc='Point cloud for wrist camera left viewpoint.'
                            #     ),
                            #     'colors': tfds.features.Tensor(
                            #         shape=((IMAGE_RES[0]*IMAGE_RES[1]), 3),
                            #         dtype=np.uint8,
                            #         doc='Colors for point cloud of wrist camera left viewpoint.'
                            #     )
                            # }),
                            'cartesian_position': tfds.features.Tensor(
                                shape=(6,),
                                dtype=np.float64,
                                doc='Robot Cartesian state',
                            ),
                            'gripper_position': tfds.features.Tensor(
                                shape=(1,),
                                dtype=np.float64,
                                doc='Gripper position statae',
                            ),
                            'joint_position': tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float64,
                                doc='Joint position state'
                            )
                        }),
                        'action_dict': tfds.features.FeaturesDict({
                            'cartesian_position': tfds.features.Tensor(
                                shape=(6,),
                                dtype=np.float64,
                                doc='Commanded Cartesian position'
                            ),
                            'cartesian_velocity': tfds.features.Tensor(
                                shape=(6,),
                                dtype=np.float64,
                                doc='Commanded Cartesian velocity'
                            ),
                            'gripper_position': tfds.features.Tensor(
                                shape=(1,),
                                dtype=np.float64,
                                doc='Commanded gripper position'
                            ),
                            'gripper_velocity': tfds.features.Tensor(
                                shape=(1,),
                                dtype=np.float64,
                                doc='Commanded gripper velocity'
                            ),
                            'joint_position': tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float64,
                                doc='Commanded joint position'
                            ),
                            'joint_velocity': tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float64,
                                doc='Commanded joint velocity'
                            )
                        }),
                        'action': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float64,
                            doc='Robot action, consists of [6x joint velocities, \
                                1x gripper position].',
                        ),
                        'discount': tfds.features.Scalar(
                            dtype=np.float32,
                            doc='Discount if provided, default to 1.'
                        ),
                        'reward': tfds.features.Scalar(
                            dtype=np.float32,
                            doc='Reward if provided, 1 on final step for demos.'
                        ),
                        'is_first': tfds.features.Scalar(
                            dtype=np.bool_,
                            doc='True on first step of the episode.'
                        ),
                        'is_last': tfds.features.Scalar(
                            dtype=np.bool_,
                            doc='True on last step of the episode.'
                        ),
                        'is_terminal': tfds.features.Scalar(
                            dtype=np.bool_,
                            doc='True on last step of the episode if it is a terminal step, True for demos.'
                        ),
                    }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'recording_folderpath': tfds.features.Text(
                        doc='Path to the folder of recordings.'
                    ),
                    'cam_extrinsics': tfds.features.FeaturesDict({
                        'wrist_left': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Extrinsics for wrist camera left viewpoint.'
                        ),
                        'wrist_right': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Extrinsics for wrist camera right viewpoint.'
                        ),
                        'ext1_left': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Extrinsics for exterior camera 1 left viewpoint.'
                        ),
                        'ext1_right': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Extrinsics for exterior camera 1 right viewpoint.'
                        ),
                        'ext2_left': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Extrinsics for exterior camera 2 left viewpoint.'
                        ),
                        'ext2_right': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Extrinsics for exterior camera 2 right viewpoint.'
                        )
                    }),
                    'cam_intrinsics': tfds.features.FeaturesDict({
                        'wrist': tfds.features.Tensor(
                            shape=(3, 3),
                            dtype=np.float32,
                            doc='Intrinsic matrix for wrist camera.'
                        ),
                        'ext1': tfds.features.Tensor(
                            shape=(3, 3),
                            dtype=np.float32,
                            doc='Intrinsic matrix for exterior camera 1.'
                        ),
                        'ext2': tfds.features.Tensor(
                            shape=(3, 3),
                            dtype=np.float32,
                            doc='Intrinsic matrix for exterior camera 2.'
                        )
                    }),
                }),
            }))

    def _split_paths(self, perc = 100):
        """Define data splits."""
        # create list of all examples -- by default we put all examples in 'train' split
        # add more elements to the dict below if you have more splits in your data
        print("Crawling all episode paths...")
        episode_paths = crawler(DATA_PATH)
        if perc < 100:
            n = max(1, int(len(episode_paths) * perc / 100))
            episode_paths = random.sample(episode_paths, n)
        print(f"loading {len(episode_paths)} episodes from {DATA_PATH}...")
        episode_paths = [p for p in episode_paths if os.path.exists(p + '/trajectory.h5') and \
                         os.path.exists(p + '/recordings/MP4')]
        print(f"Found {len(episode_paths)} episodes!")
        return {
            'train': episode_paths,
        }

# if __name__ == '__main__':
#     print("Crawling all episode paths...")
#     episode_paths = crawler(DATA_PATH)
#     episode_paths = [p for p in episode_paths if os.path.exists(p + '/trajectory.h5') and \
#                         os.path.exists(p + '/recordings/MP4')]
#     print(f"Found {len(episode_paths)} episodes!")
#     # res = _generate_examples(["/vault/CHORDSkills/DROID_RAW/CLVR/success/2023-05-09/Tue_May__9_01:34:10_2023/"])
#     res = _generate_examples(episode_paths)
#     print("Example generation complete.")

