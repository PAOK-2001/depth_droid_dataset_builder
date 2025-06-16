
import os
import glob
import json
import tqdm
import random
import argparse
import tensorflow as tf

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and/or summarize DROID episodes.")
    parser.add_argument('--download', action='store_true', help='Download episodes from GCS')
    parser.add_argument('--out', type=str, default='/vault/CHORDSkills/DROID_RAW_UPDATED/', help='Output directory for downloaded episodes')
    parser.add_argument('--ext_dir', type=str, default='/vault/CHORDSkills/DROID_UPDATED_EXTRINSICS/droid', help='Directory containing extrinsic data')
    parser.add_argument('--summarize', action='store_true', help='Summarize downloaded episodes')
    args = parser.parse_args()

    download = args.download
    summarize = args.summarize
    
    BUCKET_PREFIX = "gs://gresearch/robotics/droid_raw/1.0.1/"
    CLEANED_EXT_DIR = args.ext_dir
    BASE_EXT = os.path.join(CLEANED_EXT_DIR, 'cam2base_extrinsic_superset.json')
    OUT = args.out
    if not os.path.exists(OUT):
        os.makedirs(OUT, exist_ok=True)

    with open(BASE_EXT, 'r') as f:
        base_ext = json.load(f)
        
    if not (download or summarize):
        print("Please specify at least one of --download or --summarize.")
        exit(1)
        
    if download:
        downloaded = 0
        episode_paths = tf.io.gfile.glob("gs://gresearch/robotics/droid_raw/1.0.1/*/success/*/*/metadata_*.json")
        eps_to_download = []
        for p in tqdm.tqdm(episode_paths, desc="Processing episodes"):
            episode_id = p[:-5].split("/")[-1].split("_")[-1]
            if episode_id not in base_ext:
                continue
            eps_to_download.append(p)
            
        random.seed(42)
        PERC_TO_DOWNLOAD = 0.6

        if PERC_TO_DOWNLOAD < 1.0:
            eps_to_download = random.sample(eps_to_download, int(len(eps_to_download) * PERC_TO_DOWNLOAD))
        print(f"Found {len(eps_to_download)} episodes to download.")
        # Download episodes in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(tqdm.tqdm(executor.map(download_episode, eps_to_download), total=len(eps_to_download), desc="Downloading episodes"))
            downloaded = sum(results)
    
    # List folders in OUT
    if summarize:
        downloaded_meta = glob.glob(os.path.join(OUT, '*', 'success', '*', '*', 'metadata_*.json'))
        
        task_types = {}
        for json_file in tqdm.tqdm(downloaded_meta, desc="Processing downloaded folders"):
            episode_id = json_file[:-5].split("/")[-1].split("_")[-1]
            episode_path = os.path.dirname(json_file)
            rel_path = os.path.relpath(episode_path, OUT)
            # Perform sanity checks! 
            # 1) Make sure the episode_id is in base_ext
            if episode_id not in base_ext:
                # breakpoint()
                print(f"\033[91mEpisode {episode_path} not found in base_ext, skipping.\033[0m")
                continue
            
            # 2) Make sure the episode has recordings
            recording_path = os.path.join(episode_path, 'recordings')
            mp4_path = os.path.join(recording_path, 'MP4')
            if not os.path.exists(recording_path) or len(os.listdir(mp4_path)) == 0:
                print(f"\033[91m[ WARNING ] Episode {episode_path} has no recordings! Skipping.\033[0m")
                continue
            
            # Get task from metadata
            with open(json_file, 'r') as f:
                metadata = json.load(f)
            task = metadata.get('current_task', None)
            # take only first word and lowercase it
            if task is not None:
                task = task.split()[0].lower()
                if task not in task_types:
                    task_types[task] = 0
                else:
                    task_types[task] += 1
                    
        # Make summary histogram
        import matplotlib.pyplot as plt

        print("Task types found in downloaded episodes:")
        for task, count in task_types.items():
            print(f"{task}: {count} episodes")

        # Bar plot
        plt.figure(figsize=(10, 6))
        plt.bar(task_types.keys(), task_types.values(), color='skyblue')
        plt.xlabel('Task Type')
        plt.ylabel('Number of Episodes')
        plt.title('Number of Episodes per Task Type')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(OUT, 'task_types_histogram.png'))
            