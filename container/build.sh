#!/usr/bin/env bash
set -euo pipefail

DOCKERFILE="container/Dockerfile"
IMAGE_TAG="${1:-droid_container}"
USER_ID=$(id -u)
GROUP_ID=$(id -g)
USERNAME="dev"

echo "Using Dockerfile: $DOCKERFILE"
echo "Building Docker image with tag: $IMAGE_TAG"
echo "Using build args: USER_ID=$USER_ID, GROUP_ID=$GROUP_ID, USERNAME=$USERNAME"

docker build \
  -f "$DOCKERFILE" \
  -t "$IMAGE_TAG" \
  --build-arg USER_ID="$USER_ID" \
  --build-arg GROUP_ID="$GROUP_ID" \
  --build-arg USERNAME="$USERNAME" \
  .