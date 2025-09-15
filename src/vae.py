import torch, torch.nn as nn, torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, z_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(True),   # 128 -> 64
            nn.Conv2d(32,64, 4, 2, 1), nn.ReLU(True),   # 64  -> 32
            nn.Conv2d(64,128,4, 2, 1), nn.ReLU(True),   # 32  -> 16
            nn.Conv2d(128,256,4,2,1), nn.ReLU(True),    # 16  -> 8
        )
        self.flatten = nn.Flatten()
        self.fc_mu   = nn.Linear(256*8*8, z_dim)
        self.fc_logv = nn.Linear(256*8*8, z_dim)

    def forward(self, x):
        h = self.flatten(self.conv(x))
        return self.fc_mu(h), self.fc_logv(h)

class Decoder(nn.Module):
    def __init__(self, z_dim=32):
        super().__init__()
        self.fc = nn.Linear(z_dim, 256*8*8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), nn.ReLU(True),  # 8  -> 16
            nn.ConvTranspose2d(128,64, 4,2,1), nn.ReLU(True),  # 16 -> 32
            nn.ConvTranspose2d(64, 32, 4,2,1), nn.ReLU(True),  # 32 -> 64
            nn.ConvTranspose2d(32, 1,  4,2,1), nn.Sigmoid(),   # 64 -> 128
        )

    def forward(self, z):
        h = self.fc(z).view(-1,256,8,8)
        return self.deconv(h)

class VAE(nn.Module):
    def __init__(self, z_dim=32):
        super().__init__()
        self.enc = Encoder(z_dim)
        self.dec = Decoder(z_dim)

    def encode(self, x):
        mu, logv = self.enc(x)
        std = (0.5*logv).exp()
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z, mu, logv

    def forward(self, x):
        z, mu, logv = self.encode(x)
        xhat = self.dec(z)
        return xhat, mu, logv

def elbo_loss(x, xhat, mu, logv, beta=1.0):
    recon = F.mse_loss(xhat, x, reduction='mean')
    kl = -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())
    return recon + beta*kl, recon, kl
