import os, argparse, torch
import numpy as np
import matplotlib.pyplot as plt
from data_oasis import make_loaders
from vae import VAE
from umap import UMAP   # pip install umap-learn

@torch.no_grad()
def encode_mu(dl, model, device):
    mus = []
    for x in dl:
        x = x.to(device)
        mu, logv = model.enc(x)
        mus.append(mu.cpu())
    return torch.cat(mus, dim=0).numpy()

def plot_umap(mus, out_png):
    emb = UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean").fit_transform(mus)
    plt.figure(figsize=(5,5))
    plt.scatter(emb[:,0], emb[:,1], s=2, alpha=0.6)
    plt.title("UMAP of VAE latent means (μ)")
    plt.tight_layout(); plt.savefig(out_png, dpi=200)

@torch.no_grad()
def plot_manifold(model, out_png, grid=20, span=3.0, z_dim=2, size=128, device="cpu"):
    if z_dim != 2: 
        print("Manifold grid only for z_dim=2; skipping.")
        return
    lin = np.linspace(-span, span, grid)
    canvas = np.zeros((grid*size, grid*size), dtype=np.float32)
    for i, yi in enumerate(lin):
        for j, xj in enumerate(lin):
            z = torch.tensor([[xj, yi]], dtype=torch.float32, device=device)
            xhat = model.dec(z).cpu().numpy()[0,0]
            canvas[i*size:(i+1)*size, j*size:(j+1)*size] = xhat
    plt.figure(figsize=(6,6))
    plt.imshow(canvas, cmap="gray")
    plt.axis("off"); plt.tight_layout(); plt.savefig(out_png, dpi=250)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt", default="runs/oasis-vae-128d32/best.pt")
    ap.add_argument("--size", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    z_dim = ckpt["args"]["z_dim"]
    model = VAE(z_dim).to(device); model.load_state_dict(ckpt["model"]); model.eval()

    _, val_dl, _ = make_loaders(args.data_root, size=args.size, bs=128, workers=2)
    mus = encode_mu(val_dl, model, device)
    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    plot_umap(mus, out_png=os.path.join(os.path.dirname(args.ckpt), "umap_mu.png"))
    plot_manifold(model, out_png=os.path.join(os.path.dirname(args.ckpt), "manifold_grid.png"),
                  z_dim=z_dim, size=args.size, device=device)
