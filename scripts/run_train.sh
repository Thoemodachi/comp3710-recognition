#!/bin/bash
#SBATCH --job-name=oasis-train
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
mkdir -p logs

module purge
module load cuda
module load miniconda3   # only if required on Rangpur

source ~/miniconda3/etc/profile.d/conda.sh
conda activate recognition

python -m src.train \
  --data_root /home/groups/comp3710/OASIS \
  --out_dir runs/oasis-vae-128d32 \
  --size 128 \
  --z_dim 32 \
  --epochs 40 \
  --amp
