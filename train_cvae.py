"""
train_cvae.py

Conditional Variational Autoencoder (cVAE) for gravitational lens images.

Architecture:
  - Encoder: 4x Conv2d (stride-2) -> FC -> mu/logvar (latent_dim=32)
  - Decoder: FC -> 4x ConvTranspose2d -> Tanh
  - Condition: 4-dim vector projected to 32-dim, concatenated to encoder/decoder inputs
  - Loss: ELBO = MSE + beta * KL,  beta=4

Input:  output/v2/images.npy     (N, 125, 125) float32
        output/v2/theta_Es.npy   (N,)
        output/v2/z_lens.npy     (N,)
        output/v2/z_source.npy   (N,)
        output/v2/lensed.npy     (N,)

Output: output/v2/cvae_weights.pt       — trained model state dict
        output/v2/cvae_train_log.npy    — (epoch, train_loss, val_loss) per epoch

Usage:
    pip install torch torchvision
    python train_cvae.py
    python train_cvae.py --data output/v2 --epochs 100 --batch 32 --latent 32 --beta 4
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(latent_dim, cond_dim, image_size):
    """Returns (encoder, decoder) as a single nn.Module via CVAE class."""
    import torch
    import torch.nn as nn

    class CVAE(nn.Module):
        def __init__(self):
            super().__init__()

            # ── Encoder ───────────────────────────────────────────────────
            # Input: (B, 1, 128, 128)  [padded from 125x125]
            # Each Conv halves spatial dims: 128->64->32->16->8
            self.enc_conv = nn.Sequential(
                nn.Conv2d(1,   32,  4, stride=2, padding=1),  # -> (B,32,64,64)
                nn.LeakyReLU(0.2),
                nn.Conv2d(32,  64,  4, stride=2, padding=1),  # -> (B,64,32,32)
                nn.LeakyReLU(0.2),
                nn.Conv2d(64,  128, 4, stride=2, padding=1),  # -> (B,128,16,16)
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 256, 4, stride=2, padding=1),  # -> (B,256,8,8)
                nn.LeakyReLU(0.2),
            )
            enc_flat = 256 * 8 * 8  # = 16384

            # Condition projection
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_dim, 32),
                nn.LeakyReLU(0.2),
            )

            # mu/logvar heads
            self.fc_mu     = nn.Linear(enc_flat + 32, latent_dim)
            self.fc_logvar = nn.Linear(enc_flat + 32, latent_dim)

            # ── Decoder ───────────────────────────────────────────────────
            self.dec_fc = nn.Sequential(
                nn.Linear(latent_dim + 32, 256 * 4 * 4),
                nn.ReLU(),
            )
            # Each ConvTranspose2d doubles spatial dims: 4->8->16->32->64->128
            self.dec_conv = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 4->8
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1),  # 8->16
                nn.ReLU(),
                nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1),  # 16->32
                nn.ReLU(),
                nn.ConvTranspose2d(32,  16,  4, stride=2, padding=1),  # 32->64
                nn.ReLU(),
                nn.ConvTranspose2d(16,   1,  4, stride=2, padding=1),  # 64->128
                nn.Tanh(),
            )

        def encode(self, x, cond):
            h = self.enc_conv(x)              # (B, 256, 8, 8)
            h = h.view(h.size(0), -1)         # (B, 16384)
            c = self.cond_proj(cond)           # (B, 32)
            hc = torch.cat([h, c], dim=1)     # (B, 16416)
            return self.fc_mu(hc), self.fc_logvar(hc)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z, cond):
            c  = self.cond_proj(cond)          # (B, 32)
            zc = torch.cat([z, c], dim=1)      # (B, latent+32)
            h  = self.dec_fc(zc)               # (B, 256*4*4)
            h  = h.view(h.size(0), 256, 4, 4) # (B, 256, 4, 4)
            return self.dec_conv(h)            # (B, 1, 128, 128)

        def forward(self, x, cond):
            mu, logvar = self.encode(x, cond)
            z   = self.reparameterize(mu, logvar)
            out = self.decode(z, cond)
            return out, mu, logvar

        def sample(self, cond, n=1, device='cpu'):
            """Generate n samples conditioned on cond (shape: (n, cond_dim))."""
            self.eval()
            with torch.no_grad():
                z = torch.randn(n, latent_dim, device=device)
                return self.decode(z, cond)

    return CVAE()


# ── Loss ──────────────────────────────────────────────────────────────────────

def elbo_loss(recon, target, mu, logvar, beta):
    import torch.nn.functional as F
    mse = F.mse_loss(recon, target, reduction='sum') / target.size(0)
    kl  = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    return mse + beta * kl, mse, kl


# ── Data prep ─────────────────────────────────────────────────────────────────

def load_data(data_dir, image_size=128):
    """
    Load and preprocess simulation outputs.

    Returns X (N, 1, 128, 128) in [-1, 1] and cond (N, 4).
    """
    print(f'Loading data from {data_dir}/')
    images   = np.load(os.path.join(data_dir, 'images.npy'))    # (N, 125, 125)
    theta_Es = np.load(os.path.join(data_dir, 'theta_Es.npy'))  # (N,)
    z_lens   = np.load(os.path.join(data_dir, 'z_lens.npy'))    # (N,)
    z_source = np.load(os.path.join(data_dir, 'z_source.npy'))  # (N,)
    lensed   = np.load(os.path.join(data_dir, 'lensed.npy'))    # (N,)

    N = len(images)
    print(f'  {N} images  shape={images.shape}')

    # Pad 125x125 -> 128x128 (power-of-2 required for stride-2 convolutions)
    # Reflect padding preserves background statistics at borders
    images = np.pad(images, ((0, 0), (1, 2), (1, 2)), mode='reflect')
    assert images.shape[1:] == (image_size, image_size), images.shape

    # sqrt stretch (compresses dynamic range, emphasizes arcs)
    images = np.sqrt(np.clip(images, 0, None))

    # Normalize to [-1, 1] globally across the dataset
    vmin = images.min()
    vmax = images.max()
    images = 2.0 * (images - vmin) / (vmax - vmin + 1e-9) - 1.0
    images = images[:, np.newaxis, :, :].astype(np.float32)  # (N, 1, 128, 128)

    # Condition vector: normalize to roughly [-1, 1] / [0, 1]
    # theta_E / 2.0,  z_lens / 0.9,  z_source / 3.0,  lensed_label
    cond = np.stack([
        theta_Es / 2.0,
        z_lens   / 0.9,
        z_source / 3.0,
        lensed.astype(np.float32),
    ], axis=1).astype(np.float32)   # (N, 4)

    print(f'  images range: [{images.min():.3f}, {images.max():.3f}]')
    print(f'  cond  shape : {cond.shape}')
    return images, cond


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    import torch
    from torch.utils.data import TensorDataset, DataLoader, random_split

    # ── Device ────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using device: MPS (Apple Silicon)')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using device: CUDA ({torch.cuda.get_device_name(0)})')
    else:
        device = torch.device('cpu')
        print('Using device: CPU')

    # ── Data ──────────────────────────────────────────────────────────────
    images, cond = load_data(args.data)
    X_t = torch.tensor(images)
    C_t = torch.tensor(cond)

    dataset = TensorDataset(X_t, C_t)
    n_val   = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                               num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                               num_workers=0, pin_memory=(device.type == 'cuda'))
    print(f'Train: {n_train}  Val: {n_val}  Batch: {args.batch}')

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(args.latent, cond_dim=4, image_size=128).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {n_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── Training ──────────────────────────────────────────────────────────
    out_dir = Path(args.data)
    log = []
    best_val = float('inf')

    print(f'\nTraining for {args.epochs} epochs...')
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_sum = 0.0

        for X_batch, C_batch in train_loader:
            X_batch = X_batch.to(device)
            C_batch = C_batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(X_batch, C_batch)
            loss, mse, kl = elbo_loss(recon, X_batch, mu, logvar, args.beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss_sum += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for X_batch, C_batch in val_loader:
                X_batch = X_batch.to(device)
                C_batch = C_batch.to(device)
                recon, mu, logvar = model(X_batch, C_batch)
                loss, _, _ = elbo_loss(recon, X_batch, mu, logvar, args.beta)
                val_loss_sum += loss.item()

        train_loss = train_loss_sum / len(train_loader)
        val_loss   = val_loss_sum   / len(val_loader)
        log.append([epoch, train_loss, val_loss])

        elapsed = time.time() - t0
        if epoch % 10 == 0 or epoch <= 5:
            lr_now = optimizer.param_groups[0]['lr']
            print(f'  Epoch {epoch:4d}/{args.epochs}  '
                  f'train={train_loss:.4f}  val={val_loss:.4f}  '
                  f'lr={lr_now:.2e}  {elapsed:.1f}s')

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_dir / 'cvae_weights_best.pt')

    # Save final model and log
    torch.save(model.state_dict(), out_dir / 'cvae_weights.pt')
    np.save(str(out_dir / 'cvae_train_log.npy'), np.array(log))
    print(f'\nSaved -> {out_dir}/cvae_weights.pt')
    print(f'Saved -> {out_dir}/cvae_train_log.npy')
    print(f'Best val loss: {best_val:.4f}')

    return model, device, cond


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train conditional VAE on lens images')
    parser.add_argument('--data',    default='output/v2',
                        help='Directory containing images.npy etc. (default: output/v2)')
    parser.add_argument('--epochs',  type=int,   default=100)
    parser.add_argument('--batch',   type=int,   default=32)
    parser.add_argument('--latent',  type=int,   default=32)
    parser.add_argument('--beta',    type=float, default=4.0,
                        help='KL weight in ELBO (default: 4)')
    parser.add_argument('--lr',      type=float, default=1e-4)
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        raise SystemExit('PyTorch not installed. Run: pip install torch torchvision')

    train(args)
