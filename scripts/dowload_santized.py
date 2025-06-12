
import os
import json
import tqdm
import random
import tensorflow as tf
BUCKET_PREFIX = "gs://gresearch/robotics/droid_raw/1.0.1/"
CLEANED_EXT_DIR = '/vault/CHORDSkills/DROID_UPDATED_EXTRINSICS/droid'
BASE_EXT = os.path.join(CLEANED_EXT_DIR, 'cam2base_extrinsic_superset.json')
OUT = '/vault/CHORDSkills/DROID_RAW_UPDATED/'
if not os.path.exists(OUT):
    os.makedirs(OUT, exist_ok=True)

with open(BASE_EXT, 'r') as f:
    base_ext = json.load(f)

downloaded = 0
episode_paths = tf.io.gfile.glob("gs://gresearch/robotics/droid_raw/1.0.1/*/success/*/*/metadata_*.json")
eps_to_download = []
for p in tqdm.tqdm(episode_paths, desc="Processing episodes"):
    episode_id = p[:-5].split("/")[-1].split("_")[-1]
    if episode_id not in base_ext:
        continue
    eps_to_download.append(p)
    
PERC_TO_DOWNLOAD = 0.6
random.seed(42)
if PERC_TO_DOWNLOAD < 1.0:
    eps_to_download = random.sample(eps_to_download, int(len(eps_to_download) * PERC_TO_DOWNLOAD))
print(f"Found {len(eps_to_download)} episodes to download.")

import concurrent.futures
def download_episode(p):
    base_dir = os.path.dirname(p)
    ep_folder = base_dir.replace(BUCKET_PREFIX, "")
    ep_folder = os.path.join(OUT, ep_folder)
    if not os.path.exists(ep_folder):
        os.makedirs(ep_folder, exist_ok=True)
        os.system(f'gsutil -m cp -r {base_dir}/* {ep_folder}')
        return 1
    return 0

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(tqdm.tqdm(executor.map(download_episode, eps_to_download), total=len(eps_to_download), desc="Downloading episodes"))
    downloaded = sum(results)
    
print(f"Downloaded {downloaded} episodes to {OUT}")