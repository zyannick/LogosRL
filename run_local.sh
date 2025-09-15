#!/bin/bash
set -e 

IMAGE_TAG="yzoetgna/logos-rl:v1.0"

mkdir -p ./data ./logs ./moe_outputs ./mlruns ./hf_cache ./mpl_config

COMMON_MOUNTS="-v ./data:/app/data \
               -v ./moe_outputs:/app/moe_outputs \
               -v ./mlruns:/app/mlruns \
               -v ./logs:/app/logs \
               -v ./hf_cache:/app/hf_cache \
               -v ./mpl_config:/app/mpl_config \
               -v ./.git:/app/.git"


docker run --rm \
  --gpus all \
  --user $(id -u):$(id -g) \
  -e HF_HOME=/app/hf_cache \
  -e MPLCONFIGDIR=/app/mpl_config \
  $COMMON_MOUNTS \
  $IMAGE_TAG \
  dvc "$@"