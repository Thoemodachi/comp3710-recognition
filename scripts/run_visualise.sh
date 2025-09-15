#!/bin/bash
#SBATCH --job-name=oasis-vis
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
mkdir -p logs

module purge
module load cuda
module load miniconda3   # only if required on Rangpur

source ~/miniconda3/etc/profile.d/conda.sh
conda activate recognition

python -m src.visualize \
  --data_root /home/groups/comp3710/OASIS \
  --ckpt runs/oasis-vae-128d32/best.pt \
  --size 128
