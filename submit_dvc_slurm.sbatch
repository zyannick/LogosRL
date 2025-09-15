#!/bin/bash
#SBATCH --job-name=logos-rl-training
#SBATCH --mail-user=yannickzoet@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --mem 50G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -L sps
#SBATCH --time 02-00:00:00         
#SBATCH --output=slurm_logs/%x_%j.out  



##module load cuda/12.1
module load apptainer

echo "INFO: Running DVC pipeline with Apptainer..."

SCRATCH_DIR=${SCRATCH:-"/scratch/users/${USER}"}
export APPTAINER_CACHEDIR="${SCRATCH_DIR}/apptainer_cache"
mkdir -p "$APPTAINER_CACHEDIR"
echo "INFO: Apptainer cache is set to $APPTAINER_CACHEDIR"

export HOST_PROJECT_DIR=$PWD
export HOST_DATA_DIR=$PWD/data
export HOST_OUTPUT_DIR=$PWD/moe_outputs
export HOST_HF_CACHE_DIR=$PWD/hf_cache

apptainer exec --nv \
    --bind $HOST_DATA_DIR:/app/data \
    --bind $HOST_OUTPUT_DIR:/app/moe_outputs \
    --bind $HOST_HF_CACHE_DIR:/app/hf_cache \
    --bind $PWD/.git:/app/.git \
    --env HF_HOME=/app/hf_cache \
    docker://ataraxai/logos-rl:v1.0 \
    bash -c "dvc repro run_pipeline"