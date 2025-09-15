# comp3710-recognition
## Variational Autoencoder on OASIS MRI Slices

### Overview

This project implements a Variational Autoencoder (VAE) using PyTorch to learn representations of 2D brain MRI slices from the OASIS dataset. It includes training, evaluation, and visualisation of latent structure using UMAP or a 2D manifold.

### Requirements

* Python 3.7+
* PyTorch
* torchvision
* umap-learn
* PIL (Pillow)
* numpy, matplotlib

Optionally:

* CUDA / GPU support for faster training
* Mixed precision (if enabled)

### Data

Data are expected in the following structure:

```
{data_root}/
  keras_png_slices_train/
  keras_png_slices_validate/
  keras_png_slices_test/
```

Only the slice directories (non-segmentation) are used.

### Project Structure

```
oasis-vae/
├─ src/
│   ├ data_oasis.py         # dataset loader
│   ├ vae.py                # VAE model definition, encoder, decoder
│   ├ train.py              # training loop, loss, checkpointing
│   ├ visualize.py          # UMAP and manifold visualisation
│   └ utils.py              # auxiliary utilities (seed, metric logging, etc.)
├─ scripts/                 # run scripts / batch job wrappers
├─ runs/                    # output directory for models, logs, images
├─ README.md                # this file
└─ environment.yml / requirements.txt
```

### How to Run

1. **Training**

   ```bash
   python -m src.train \
     --data_root /path/to/keras_png_slices_data \
     --out_dir runs/oasis-vae \
     --size 128 \
     --z_dim 32 \
     --batch_size 64 \
     --epochs 40 \
     --lr 1e-3
   ```

   Optional flags:
   `--amp` for mixed precision (if supported),
   `--warmup_epochs` to control β warm-up schedule.

2. **Visualisation**

   After training, generate latent visualisations:

   ```bash
   python -m src.visualize \
     --data_root /path/to/keras_png_slices_data \
     --ckpt runs/oasis-vae/best.pt
   ```

   This produces:

   * UMAP plot of latent means (μ) for validation set
   * Manifold grid if `z_dim` = 2

### Hyperparameters & Notes

* `z_dim`: dimension of latent code; typical values are 16, 32, or 64.
* Input image size: typically 128×128. Larger resolution may improve detail but increases memory/training time.
* Loss: MSE for reconstruction, KL for regularisation.
* β schedule: linear warm-up from 0 to 1 over first few epochs helps avoid posterior collapse.

### Results & Visuals

Include reconstructions vs inputs. Include the UMAP plot to show how latent space clusters. If z\_dim = 2, show manifold grid (decoded images over 2D latent grid).

### Citation & Sources

* Kingma, D. P., & Welling, M. (2013). *Auto-Encoding Variational Bayes*.
* UMAP: McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.*