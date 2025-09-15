#!/bin/bash
#SBATCH --job-name=oasis-vis
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Load modules (adjust if your site has specific module versions)
module purge
module load cuda

# Activate your conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate recognition

# Run visualisation
python -m src.visualize \
  --data_root /home/groups/comp3710/keras_png_slices_data \
  --ckpt runs/oasis-vae-128d32/best.pt \
  --size 128