# comp3710-recognition
## VAE on OASIS MRI Slices

### Setup
```bash
conda env create -f environment.yml
conda activate oasis-vae
````

### Data

```
/home/groups/comp3710/keras_png_slices_data/
  keras_png_slices_train/
  keras_png_slices_validate/
  keras_png_slices_test/
```

### Train

```bash
python -m src.train \
  --data_root /home/groups/comp3710/keras_png_slices_data \
  --out_dir runs/oasis-vae \
  --size 128 --z_dim 32 --batch_size 64 --epochs 40 --lr 1e-3
```

or with Slurm:

```bash
sbatch scripts/run_train.sbatch
```

### Visualise

```bash
python -m src.visualize \
  --data_root /home/groups/comp3710/keras_png_slices_data \
  --ckpt runs/oasis-vae/best.pt
```

or with Slurm:

```bash
sbatch scripts/run_visualise.sbatch
```