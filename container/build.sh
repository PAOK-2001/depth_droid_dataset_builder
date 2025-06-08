#!/usr/bin/env bash
set -euo pipefail

DOCKERFILE="container/Dockerfile"
IMAGE_TAG="${1:-droid_container}"
echo "Using Dockerfile: $DOCKERFILE"
echo "Building Docker image with tag: $IMAGE_TAG"
docker build -f "$DOCKERFILE" -t "$IMAGE_TAG" . 