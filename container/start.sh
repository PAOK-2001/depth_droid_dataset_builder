#!/usr/bin/env bash

xhost +local:root

docker run -it -d --rm --privileged \
  --name droid_container \
  --net=host \
  --gpus all \
  --volume="$PWD:/root/droid-example/droid_package:rw" \
  --volume="/vault/CHORDSkills/:/vault/CHORDSkills:rw" \
  --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
  -e DISPLAY=$DISPLAY \
  droid_container
