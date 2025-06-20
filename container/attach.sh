#!/usr/bin/env bash

CONTAINER_NAME="${1:-droid_container}"
USERNAME=dev
USER_ID=$(id -u)
GROUP_ID=$(id -g)

docker exec -it \
  -u $USER_ID:$GROUP_ID \
  -e HOME=/home/$USERNAME \
  -e XDG_RUNTIME_DIR=/home/$USERNAME/.xdg \
  $CONTAINER_NAME fish
