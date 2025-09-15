#!/bin/bash
#SBATCH --job-name=oasis-vae
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
mkdir -p logs

# Load modules (adjust if needed on Rangpur)
module purge
module load cuda
module load miniconda3   # only if Rangpur requires this

# Activate your env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate recognition

# Run training
python -m src.train \
  --data_root /home/groups/comp3710/keras_png_slices_data \
  --out_dir runs/oasis-vae-128d32 \
  --size 128 \
  --z_dim 32 \
  --epochs 40 \
  --amp