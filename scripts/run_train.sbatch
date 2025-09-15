# scripts/run_train.sh
#!/bin/bash
#SBATCH --job-name=oasis-vae
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
module purge
module load cuda # (or your site’s PyTorch module)
python -m src.train --data_root /home/groups/comp3710/keras_png_slices_data \
  --out_dir runs/oasis-vae-128d32 --size 128 --z_dim 32 --epochs 40 --amp