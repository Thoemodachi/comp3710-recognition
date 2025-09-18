"""Variational Autoencoder model components for brain slice reconstruction."""

import torch, torch.nn as nn, torch.nn.functional as F

class Encoder(nn.Module):
    """Convolutional encoder that maps input slices to latent parameters."""

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
        """Encode a batch of slices into mean and log-variance vectors."""

        h = self.flatten(self.conv(x))
        return self.fc_mu(h), self.fc_logv(h)

class Decoder(nn.Module):
    """Mirror decoder that upsamples latent vectors back to image space."""

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
        """Decode latent vectors into reconstructed slice tensors."""

        h = self.fc(z).view(-1,256,8,8)
        return self.deconv(h)

class VAE(nn.Module):
    """End-to-end Variational Autoencoder with reparameterised sampling."""

    def __init__(self, z_dim=32):
        super().__init__()
        self.enc = Encoder(z_dim)
        self.dec = Decoder(z_dim)

    def encode(self, x):
        """Return sampled latent code along with its distribution parameters."""

        mu, logv = self.enc(x)
        std = (0.5*logv).exp()
        eps = torch.randn_like(std)
        # Reparameterisation trick: z = μ + σ ⊙ ε enables backpropagation
        z = mu + eps*std
        return z, mu, logv

    def forward(self, x):
        """Reconstruct ``x`` by sampling a latent representation and decoding."""

        z, mu, logv = self.encode(x)
        xhat = self.dec(z)
        return xhat, mu, logv

def elbo_loss(x, xhat, mu, logv, beta=1.0):
    """Compute the Evidence Lower Bound loss for a batch.

    Args:
        x: Original batch of slices.
        xhat: Reconstruction produced by the decoder.
        mu: Latent mean vector returned by the encoder.
        logv: Latent log-variance vector returned by the encoder.
        beta: Weighting factor applied to the KL divergence term.

    Returns:
        Tuple ``(loss, recon, kl)`` containing the scalar ELBO objective and its
        constituent reconstruction and KL components.
    """
    recon = F.mse_loss(xhat, x, reduction='mean')
    kl = -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())
    return recon + beta*kl, recon, kl
