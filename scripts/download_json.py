import os
import json
import tqdm
import tensorflow as tf

OUT = '/data/droid/DROID_3D/'
json_paths = tf.io.gfile.glob("gs://gresearch/robotics/droid_raw/1.0.1/*.json")
for file in tqdm.tqdm(json_paths, desc="Downloading JSON files"):
    out_path = os.path.join(OUT, os.path.basename(file))
    os.system(f'gsutil cp {file} {out_path}')