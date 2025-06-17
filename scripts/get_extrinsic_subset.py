# The OG DROID dataset has some noisy extrinsics. They released a subset of the dataset with cleaned extrinsics. This script separates the cleaned extrinsics from the original dataset.
import os
import glob
import tqdm
import json
import json, os, re, tempfile
from pathlib import Path
import shutil
from shutil import copytree
import numpy as np
from tqdm import tqdm
import numpy as np
import pandas as pd

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

def episode_id(path: Path) -> str:
        """
        Return the string that follows 'metadata_' and precedes '.json'.

        Examples
        --------
        metadata_TRI+52ca9b6a+2024-02-13-10h-46m-49s.json
        →  TRI+52ca9b6a+2024-02-13-10h-46m-49s
        """
        stem = path.stem                   # e.g. 'metadata_TRI+52ca9b6a+2024-02-13-10h-46m-49s'
        prefix = 'metadata_'
        if not stem.startswith(prefix):
            raise ValueError(f"{path.name} does not start with '{prefix}'")
        return stem[len(prefix):]   

def get_ext_subset(
        threshold: bool = True,
        thrs: dict = None,
        in_place: bool = False,
        auto_quantile: float = 0.60
    ):
    """
    Copy episodes whose extrinsics pass quality filters into OUT_DIR
    and overwrite their metadata with the cleaned extrinsics.

    Args
    ----
    threshold      : If True, apply quality thresholds.
    thrs           : Dict of metric → threshold.  If None they are
                     derived automatically from BASE_EXT and `auto_quantile`.
    in_place       : If True, patch metadata in RAW_DIR instead of copying.
    auto_quantile  : Used only when `thrs is None`.
    """
    # ---------------- Paths --------------------------------------------------
    RAW_DIR = Path('/vault/CHORDSkills/DROID_RAW_UPDATED')
    OUT_DIR = Path('/vault/CHORDSkills/DROID_3D')
    BASE_EXT = Path(os.path.join(CLEANED_EXT_DIR, 'cam2base_extrinsic_superset.json'))
    
    # ---------------- helpers -------------------------------------------------
    def passes_threshold(ep_info: dict) -> bool:
        if not threshold:                     # fast path
            return True
        for ser in (k for k in ep_info if k.isdigit()):
            metric = ep_info[f'{ser}_metric_type']
            val    = ep_info[f'{ser}_quality_metric']
            chk    = 1/val if metric == 'Reprojection_error' else val
            if chk < thrs[metric]:
                return False
        return True

    def patch_metadata(meta_path: Path, extrinsics: dict):
        with open(meta_path) as f:
            meta = json.load(f)
        for cam in ['wrist', 'ext1', 'ext2']:
            ser = meta[f'{cam}_cam_serial']
            if ser in extrinsics:
                meta[f'{cam}_cam_extrinsics'] = extrinsics[ser]
        # atomic replace → never leaves half-written JSONs
        with tempfile.NamedTemporaryFile('w', delete=False,
                                         dir=meta_path.parent) as tf:
            json.dump(meta, tf, indent=4)
            tmp = tf.name
        os.replace(tmp, meta_path)
        
    def has_recording(episode_path: Path) -> bool:
        recording_path = os.path.join(episode_path, 'recordings')
        mp4_path = os.path.join(recording_path, 'MP4')
        if not os.path.exists(recording_path) or len(os.listdir(mp4_path)) == 0:
            print(f"\033[91m[ WARNING ] Episode {episode_path} has no recordings! Skipping.\033[0m")
            return False
        
        # If MP4 path exits make sure there is at least one recording with stero X-stereo
        mp4_files = glob.glob(os.path.join(mp4_path, '*-stereo.mp4'))
        if len(mp4_files) == 0:
            print(f"\033[91m[ WARNING ] Episode {episode_path} has no stereo recordings! Skipping.\033[0m")
            return False
        return True
            

    # ---------------- derive thresholds if needed ----------------------------
    with open(BASE_EXT) as f:
        base_ext = json.load(f)

    # ---------------- ensure output location ---------------------------------
    if not in_place:
        if OUT_DIR.exists():
            shutil.rmtree(OUT_DIR)
        OUT_DIR.mkdir(parents=True)

    # ---------------- crawl & copy/patch -------------------------------------
    kept = 0
    episode_list = glob.glob(os.path.join(RAW_DIR, '*', 'success', '*', '*', 'metadata_*.json'))
    tri_episodes = []
    for ep in episode_list:
        if '_TRI+' in ep:
            tri_episodes.append(ep)
    for ep_json in tqdm(episode_list, desc='Processing episodes'):
        ep_json = Path(ep_json)
        ep_id = episode_id(ep_json)
        ext_info = base_ext.get(ep_id)
        if ext_info is None or not passes_threshold(ext_info) or not has_recording(ep_json.parent):
            continue

        dst_dir = ep_json.parent if in_place else \
                  OUT_DIR / ep_json.relative_to(RAW_DIR).parent

        if not in_place:                     # copy whole episode folder
            copytree(ep_json.parent, dst_dir, dirs_exist_ok=False)
            # NOTE: some episodes have two metadata files, make sure to only keep the one that matches the base_ext
            metadata_files = list(dst_dir.glob('metadata_*.json'))
            # Delete all metadata files except the one that matches the base_ext
            for meta_file in metadata_files:
                if meta_file.name != ep_json.name:
                    print(f"Found duplicate metadata file: {meta_file.name}, "
                          f"deleting it.")
                    meta_file.unlink()
            
        assert ep_id in base_ext, \
            f"Episode {ep_id} not found in base_ext."
        patch_metadata(dst_dir / ep_json.name, ext_info)
        kept += 1

    print(f"✅  {kept} episodes copied/updated "
          f"({'in-place' if in_place else 'to OUT_DIR'}).")
    
        
def sanity_check():
    with open(BASE_EXT, 'r') as f:
        base_ext = json.load(f)
    # Check if the cleaned extrinsics are correct
    metrics = {}
    episode_list = glob.glob(os.path.join(OUT_DIR, '*', 'success', '*', '*', 'metadata_*.json'))
    for episode in tqdm(episode_list, desc="Sanity checking episodes"):
        ep_id = episode_id(Path(episode))
        if ep_id not in base_ext.keys():
            continue
        
        with open(episode, 'r') as f:
            metadata = json.load(f)
        for cam in ['wrist', 'ext1', 'ext2']:
            serial = metadata[f'{cam}_cam_serial']
            if serial in base_ext[ep_id].keys():
                ext = base_ext[ep_id][serial]
                if not np.allclose(metadata[f'{cam}_cam_extrinsics'], ext, atol=1e-6):
                    raise ValueError(f"Episode {ep_id} has incorrect extrinsics for {cam} camera.")
                metric, val = get_cam_metrics(base_ext[ep_id], serial)
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