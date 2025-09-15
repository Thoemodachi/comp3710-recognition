import os, argparse, torch
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from src import make_loaders, VAE, elbo_loss

def set_seed(seed=1337):
    import random, numpy as np
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def train(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl, val_dl, test_dl = make_loaders(args.data_root, size=args.size,
                                             bs=args.batch_size, workers=args.workers)
    model = VAE(args.z_dim).to(device)
    opt   = Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=args.amp)

    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    best_val = 1e9

    def beta(epoch):
        if epoch < args.warmup_epochs:
            return (epoch+1)/max(1,args.warmup_epochs) * args.beta_end
        return args.beta_end

    for epoch in range(args.epochs):
        model.train(); tr_loss=tr_rec=tr_kl=0.0
        for x in train_dl:
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp):
                xhat, mu, logv = model(x)
                b = beta(epoch)
                loss, rec, kl = elbo_loss(x, xhat, mu, logv, beta=b)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tr_loss += loss.item(); tr_rec += rec.item(); tr_kl += kl.item()
        trn = len(train_dl); tr_stats = (tr_loss/trn, tr_rec/trn, tr_kl/trn)

        # validation
        model.eval(); vl_loss=vl_rec=vl_kl=0.0
        with torch.no_grad():
            for x in val_dl:
                x = x.to(device)
                xhat, mu, logv = model(x)
                loss, rec, kl = elbo_loss(x, xhat, mu, logv, beta=beta(epoch))
                vl_loss += loss.item(); vl_rec += rec.item(); vl_kl += kl.item()
        vn = len(val_dl); vl_stats = (vl_loss/vn, vl_rec/vn, vl_kl/vn)

        print(f"[{epoch+1:03d}/{args.epochs}] "
              f"train loss={tr_stats[0]:.4f} rec={tr_stats[1]:.4f} kl={tr_stats[2]:.4f} | "
              f"val loss={vl_stats[0]:.4f} rec={vl_stats[1]:.4f} kl={vl_stats[2]:.4f}")

        # checkpoint
        ckpt = {"epoch": epoch+1, "model": model.state_dict(), "args": vars(args)}
        torch.save(ckpt, outdir/"last.pt")
        if vl_stats[0] < best_val:
            best_val = vl_stats[0]
            torch.save(ckpt, outdir/"best.pt")

    print(f"Best val loss: {best_val:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True,
                   help="Dir containing keras_png_slices_{train,validate,test}")
    p.add_argument("--out_dir", type=str, default="runs/oasis-vae")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--z_dim", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta_end", type=float, default=1.0)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--amp", action="store_true")
    args = p.parse_args()
    train(args)
