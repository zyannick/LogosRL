#!/bin/bash
set -e 

IMAGE_TAG="moe-rl-app"

PIPELINE_STAGE=$1
EXTRA_DOCKER_ARGS=""

mkdir -p ./data ./logs ./moe_outputs ./mlruns hf_cache


COMMON_MOUNTS="-v ./data:/app/data \
               -v ./moe_outputs:/app/moe_outputs \
               -v ./mlruns:/app/mlruns \
               -v ./logs:/app/logs \
               -v ./hf_cache:/app/hf_cache \
               -v ./mpl_config:/app/mpl_config \
               -v ./.git:/app/.git"


if [ "$PIPELINE_STAGE" == "prepare_data" ]; then
    EXTRA_DOCKER_ARGS="$COMMON_MOUNTS"
elif [ "$PIPELINE_STAGE" == "full_pipeline" ]; then
    EXTRA_DOCKER_ARGS="--gpus all $COMMON_MOUNTS"
else
    echo "Error: Unknown pipeline stage '$PIPELINE_STAGE'"
    exit 1
fi

docker run --rm \
  --user $(id -u):$(id -g) \
  -e HF_HOME=/app/hf_cache \
  -e MPLCONFIGDIR=/app/mpl_config \
  $EXTRA_DOCKER_ARGS \
  $IMAGE_TAG \
  torchrun --nproc_per_node=auto src/run_pipeline.py --pipeline_stage $PIPELINE_STAGE