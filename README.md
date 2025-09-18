# OASIS VAE Workflow

## 1. Environment
```bash
conda env create -f environment.yml
conda activate oasis-vae
```

## 2. Data Layout
```
<data_root>/
  keras_png_slices_train/
  keras_png_slices_validate/
  keras_png_slices_test/
```

## 3. Train
```bash
python -m src.train \
  --data_root <data_root> \
  --out_dir runs/oasis-vae \
  --size 128 --z_dim 32 --batch_size 64 --epochs 40 --lr 1e-3
```

or submit to Slurm:
```bash
sbatch scripts/run_train
```

## 4. Visualise
```bash
python -m src.visualise \
  --data_root <data_root> \
  --ckpt runs/oasis-vae/best.pt
```

or submit to Slurm:
```bash
sbatch scripts/run_visualise
```
