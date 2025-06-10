# DROID Dataset Conversion

This forks integrated the camera calibration parameters and depth information into the packaging of the DROID dataset for applications that requiere it.

## Installation

Follow the instructions on the DROID dataset website to download the raw version [`DROID Dataset`](https://droid-dataset.github.io/droid/the-droid-dataset.html)

### Docker 
Based on ReRuns example for reading the raw DROID dataset, we adapt Docker containers for easily managing the ZED camera library.
```
chmod +x container/*
./container/build.sh
/container/start.sh
```

To use the script, connect interactively to the container.
```
./container/start.sh   
```


### Local
First create a conda environment using the provided environment.yml file (use `environment_ubuntu.yml` or `environment_macos.yml` depending on the operating system you're using):
```
conda env create -f environment_ubuntu.yml
```

Then activate the environment using:
```
conda activate rlds_env
```

## Usage
Modify the `droid_builder.py` the constant DATA_PATH to point to the raw version of the dataset.

```
tfds build --data_dir=/vault/CHORDSkills/DROID_PACKED/ droid/droid_builder.py
```