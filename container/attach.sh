#!/usr/bin/env bash

CONTAINER_NAME=droid_container
USERNAME=dev

docker exec -it --user $USERNAME \
  -e XDG_RUNTIME_DIR=/home/$USERNAME/.xdg \
  -e HOME=/home/$USERNAME \
  $CONTAINER_NAME fish
