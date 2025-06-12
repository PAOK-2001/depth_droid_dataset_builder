# The OG DROID dataset has some noisy extrinsics. They released a subset of the dataset with cleaned extrinsics. This script separates the cleaned extrinsics from the original dataset.
import os
import glob
import tqdm
import json
import numpy as np
import pandas as pd
# PATHS
RAW_DIR = '/vault/CHORDSkills/DROID_RAW'
OUT_DIR = '/vault/CHORDSkills/DROID3D_RAW'
CLEANED_EXT_DIR = '/vault/CHORDSkills/DROID_UPDATED_EXTRINSICS/droid'
BASE_EXT = os.path.join(CLEANED_EXT_DIR, 'cam2base_extrinsic_superset.json')

def get_cam_metrics(ep_info, cam_serial):
    metric = ep_info[f'{cam_serial}_metric_type']
    val = ep_info[f'{cam_serial}_quality_metric']
    return metric, val

def get_quality_thresholds(auto_quantile=0.95):
    with open(BASE_EXT, 'r') as f:
        base_ext = json.load(f)
    metrics = {}
    thrsholds = {}
    for ep, ep_info in base_ext.items():
        # breakpoint()
        # Extract keys that correspond to cam serials (numbers as strings)
        cam_serial_keys = [k for k in ep_info.keys() if k.isdigit()]
        for cam_serial in cam_serial_keys:
            metric, val = get_cam_metrics(ep_info, cam_serial)
            if metric not in metrics.keys():
                metrics[metric] = []
            metrics[metric].append(val)
 
    # Print metrics and calculate thresholds
    for metric, values in metrics.items():
        values = np.array(values)
        if metric == "Reprojection_error":
            thrsholds[metric] = 1 / float(np.quantile(values, 1- auto_quantile))
        else:
            thrsholds[metric] = float(np.quantile(values, auto_quantile))
        
        print(f"Metric: {metric}")
        print(f"  Mean: {np.mean(values)}")
        print(f"  Std: {np.std(values)}")
        print(f"  Min: {np.min(values)}")
        print(f"  Max: {np.max(values)}")
        print(f"  Threshold: {thrsholds[metric]}")
    return thrsholds

# Crawl
def get_ext_subset(threshold: bool = False, thrs = None):
    # Copy over JSON files from  RAW_DIR to OUT_DIR
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    json_files = glob.glob(os.path.join(RAW_DIR, '*.json'))
    for json_file in tqdm.tqdm(json_files, desc="Copying JSON files"):
        rel_path = os.path.relpath(json_file, RAW_DIR)
        out_path = os.path.join(OUT_DIR, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if not os.path.exists(out_path):
            os.system(f'cp {json_file} {out_path}')
    
    # Crawl over the raw directory
    episode_list = glob.glob(os.path.join(RAW_DIR, '*', 'success', '*', '*', 'metadata_*.json'))
    with open(BASE_EXT, 'r') as f:
        base_ext = json.load(f)
    
    episodes_to_keep = set(base_ext.keys())
    processed_ep = 0
    print(f"Found {len(episodes_to_keep)} episodes with cleaned extrinsics.")
    for episode in tqdm.tqdm(episode_list, desc="Processing episodes"):
        episode_id = episode[:-5].split("/")[-1].split("_")[-1]
        episode_path = os.path.dirname(episode)
        rel_path = os.path.relpath(episode_path, RAW_DIR)
        if episode_id not in episodes_to_keep:
            continue
        # Make sure the calibration is within tolerance
        ext_info = base_ext[episode_id]
        keep = True
        if threshold:
            cam_serial_keys = [k for k in ext_info.keys() if k.isdigit()]
            for cam_serial in cam_serial_keys:
                metric, val = get_cam_metrics(ext_info, cam_serial)
                if metric == 'Reprojection_error': 
                    val = 1 / val
                if val < thrs[metric]:
                    keep = False
                    continue 
        if not keep:
            continue
        # Copy episode to CLEANED_EXT_DIR
        cleaned_episode_path = os.path.join(OUT_DIR, rel_path)
        # print(f"Copying episode {episode_id} to {cleaned_episode_path}")
        os.makedirs(os.path.dirname(cleaned_episode_path), exist_ok=True)
        if not os.path.exists(cleaned_episode_path):
            os.system(f'cp -r {episode_path} {cleaned_episode_path}')
        
        # Open metadata file in copied episode
        metadata_path = os.path.join(cleaned_episode_path, f'metadata_{episode_id}.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        # Update extrinsics in metadata, this should be the matching ID in ext_info
        for cam in ['wrist', 'ext1', 'ext2']:
            serial = metadata[f'{cam}_cam_serial']
            if serial in ext_info.keys():
                metadata[f'{cam}_cam_extrinsics'] = ext_info[serial]
                
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        processed_ep += 1
        
    print(f"Processed {processed_ep} episodes with cleaned extrinsics.")
    
    
def sanity_check():
    with open(BASE_EXT, 'r') as f:
        base_ext = json.load(f)
    # Check if the cleaned extrinsics are correct
    metrics = {}
    episode_list = glob.glob(os.path.join(OUT_DIR, '*', 'success', '*', '*', 'metadata_*.json'))
    for episode in tqdm.tqdm(episode_list, desc="Sanity checking episodes"):
        episode_id = episode[:-5].split("/")[-1].split("_")[-1]
        with open(episode, 'r') as f:
            metadata = json.load(f)
        for cam in ['wrist', 'ext1', 'ext2']:
            serial = metadata[f'{cam}_cam_serial']
            if serial in base_ext[episode_id].keys():
                ext = base_ext[episode_id][serial]
                if not np.allclose(metadata[f'{cam}_cam_extrinsics'], ext, atol=1e-6):
                    raise ValueError(f"Episode {episode_id} has incorrect extrinsics for {cam} camera.")
                metric, val = get_cam_metrics(base_ext[episode_id], serial)
                if metric not in metrics.keys():
                    metrics[metric] = []
                metrics[metric].append(val)

    # Print metrics
    for metric, values in metrics.items():
        values = np.array(values)
        print(f"Metric: {metric}")
        print(f"  Mean: {np.mean(values)}")
        print(f"  Std: {np.std(values)}")
        print(f"  Min: {np.min(values)}")
        print(f"  Max: {np.max(values)}")
        print(f"  Count: {len(values)}")

    
if __name__ == '__main__':
    thrs = get_quality_thresholds(auto_quantile=0.60)
    get_ext_subset(threshold=True, thrs=thrs)
    sanity_check()