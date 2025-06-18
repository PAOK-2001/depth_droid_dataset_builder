#!/usr/bin/env bash

xhost +local:root

docker run -it -d --rm --privileged \
  --name droid_container \
  --net=host \
  --gpus all \
  -u $(id -u):$(id -g) \
  -e HOME=/home/dev \
  -e XDG_RUNTIME_DIR=/home/dev/.xdg \
  --volume="$PWD:/home/dev/droid-example/droid_package:rw" \
  --volume="/vault/CHORDSkills/:/vault/CHORDSkills:rw" \
  --volume="$HOME/.Xauthority:/home/dev/.Xauthority:rw" \
  -e DISPLAY=$DISPLAY \
  -w /home/dev/droid-example/droid_package \
  droid_container
