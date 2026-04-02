"""
train_simgan.py — SimGAN refinement for JWST gravitational lens simulations.

Takes simulated images and refines them to look more realistic using a
discriminator trained on real COSMOS-Web background patches. Preserves
lensing structure via self-regularization loss.

Architecture:
  - Refiner: ResNet (input + learned perturbation), InstanceNorm, ~3.3M params
  - Discriminator: PatchGAN (70x70 receptive field), ~2.8M params
  - Loss: LSGAN + lambda_self * L1(refined, original)

Usage:
    .venv/bin/python3 train_simgan.py
    .venv/bin/python3 train_simgan.py --epochs 200 --batch 16 --bands F277W
    .venv/bin/python3 train_simgan.py --epochs 200 --batch 8 --bands all
"""

import argparse
import os
import time
import json
from pathlib import Path
from collections import deque

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Model definitions ─────────────────────────────────────────────────────────

def build_models(n_channels, device):
    import torch
    import torch.nn as nn

    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.InstanceNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.InstanceNorm2d(channels),
            )

        def forward(self, x):
            return x + self.block(x)

    class Refiner(nn.Module):
        def __init__(self, ch_in):
            super().__init__()
            # Encoder
            self.enc = nn.Sequential(
                nn.Conv2d(ch_in, 64, 7, padding=3),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.InstanceNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.InstanceNorm2d(256),
                nn.ReLU(inplace=True),
            )
            # Residual blocks
            self.res = nn.Sequential(*[ResidualBlock(256) for _ in range(6)])
            # Decoder
            self.dec = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, ch_in, 7, padding=3),
                nn.Tanh(),
            )

        def forward(self, x):
            return self.dec(self.res(self.enc(x)))

    class PatchDiscriminator(nn.Module):
        def __init__(self, ch_in):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(ch_in, 64, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, stride=2, padding=1),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 512, 4, stride=1, padding=1),
                nn.InstanceNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512, 1, 4, stride=1, padding=1),
            )

        def forward(self, x):
            return self.net(x)

    refiner = Refiner(n_channels).to(device)
    disc = PatchDiscriminator(n_channels).to(device)

    n_r = sum(p.numel() for p in refiner.parameters() if p.requires_grad)
    n_d = sum(p.numel() for p in disc.parameters() if p.requires_grad)
    print(f'Refiner: {n_r:,} params  |  Discriminator: {n_d:,} params')

    return refiner, disc


class HistoryBuffer:
    """Buffer of previously refined images to stabilize discriminator training."""
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push_and_pop(self, images):
        import torch
        result = []
        for img in images:
            if len(self.buffer) < self.max_size:
                self.buffer.append(img.clone())
                result.append(img)
            elif np.random.random() > 0.5:
                idx = np.random.randint(0, len(self.buffer))
                old = self.buffer[idx].clone()
                self.buffer[idx] = img.clone()
                result.append(old)
            else:
                result.append(img)
        return torch.stack(result)


def make_preview(sim, refined, real, epoch, out_dir, n_show=5):
    """Save a visual comparison grid: sim | refined | real."""
    fig, axes = plt.subplots(3, n_show, figsize=(3 * n_show, 9))
    titles = ['Simulated', 'Refined', 'Real']

    for col in range(n_show):
        for row, (data, title) in enumerate(zip([sim, refined, real], titles)):
            ax = axes[row, col]
            # Show first channel (or average if multi-band)
            if data.shape[1] > 1:
                img = data[col].mean(axis=0)
            else:
                img = data[col, 0]
            # Denormalize from [-1,1] to [0,1]
            img = (img + 1) / 2
            ax.imshow(img, cmap='gray', origin='lower', vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(title, fontsize=12)

    plt.suptitle(f'SimGAN Epoch {epoch}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'preview_epoch_{epoch:04d}.png'), dpi=100)
    plt.close()


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    import torch
    from torch.utils.data import TensorDataset, DataLoader

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
    sim_all = np.load(os.path.join(args.data, 'sim_train.npy'))
    real_all = np.load(os.path.join(args.data, 'real_train.npy'))

    # Band selection
    band_names = ['F115W', 'F150W', 'F277W', 'F444W']
    if args.bands == 'all':
        band_idx = list(range(4))
    else:
        band_idx = [band_names.index(args.bands)]
    n_channels = len(band_idx)

    sim_all = sim_all[:, band_idx]
    real_all = real_all[:, band_idx]
    print(f'Bands: {[band_names[i] for i in band_idx]} ({n_channels} channels)')
    print(f'Sim: {sim_all.shape}  Real: {real_all.shape}')

    sim_tensor = torch.tensor(sim_all, dtype=torch.float32)
    real_tensor = torch.tensor(real_all, dtype=torch.float32)

    sim_loader = DataLoader(
        TensorDataset(sim_tensor),
        batch_size=args.batch, shuffle=True,
        num_workers=0, pin_memory=(device.type == 'cuda')
    )
    real_loader = DataLoader(
        TensorDataset(real_tensor),
        batch_size=args.batch, shuffle=True,
        num_workers=0, pin_memory=(device.type == 'cuda')
    )

    # ── Models ────────────────────────────────────────────────────────────
    refiner, disc = build_models(n_channels, device)
    opt_r = torch.optim.Adam(refiner.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))
    history = HistoryBuffer(max_size=50)

    # ── Output ────────────────────────────────────────────────────────────
    ckpt_dir = os.path.join(args.out_dir, 'checkpoints')
    preview_dir = os.path.join(args.out_dir, 'previews')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    # Fixed samples for consistent previews
    n_preview = min(5, len(sim_tensor), len(real_tensor))
    sim_fixed = sim_tensor[:n_preview].to(device)
    real_fixed = real_tensor[:n_preview].numpy()

    # ── Training loop ─────────────────────────────────────────────────────
    log = []
    lambda_self = args.lambda_self
    t0 = time.time()

    print(f'\nTraining SimGAN: {args.epochs} epochs, batch={args.batch}, '
          f'λ_self={lambda_self}')
    print(f'Output: {args.out_dir}\n')

    for epoch in range(1, args.epochs + 1):
        refiner.train()
        disc.train()

        ep_loss_r = 0
        ep_loss_d = 0
        ep_loss_self = 0
        n_batches = 0

        real_iter = iter(real_loader)

        for (sim_batch,) in sim_loader:
            sim_batch = sim_batch.to(device)

            # Get real batch (cycle if fewer real than sim)
            try:
                (real_batch,) = next(real_iter)
            except StopIteration:
                real_iter = iter(real_loader)
                (real_batch,) = next(real_iter)
            real_batch = real_batch.to(device)

            # Random augmentation (flips)
            if np.random.random() > 0.5:
                sim_batch = torch.flip(sim_batch, [2])
                real_batch = torch.flip(real_batch, [2])
            if np.random.random() > 0.5:
                sim_batch = torch.flip(sim_batch, [3])
                real_batch = torch.flip(real_batch, [3])

            # ── Train Discriminator ───────────────────────────────────
            opt_d.zero_grad()
            with torch.no_grad():
                refined = refiner(sim_batch)
            refined_hist = history.push_and_pop(refined.detach()).to(device)

            pred_real = disc(real_batch)
            pred_fake = disc(refined_hist)

            loss_d = (torch.mean((pred_real - 1) ** 2) +
                      torch.mean(pred_fake ** 2)) * 0.5

            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=5.0)
            opt_d.step()

            # ── Train Refiner ─────────────────────────────────────────
            opt_r.zero_grad()
            refined = refiner(sim_batch)
            pred_refined = disc(refined)

            loss_adv = torch.mean((pred_refined - 1) ** 2)
            loss_self = torch.mean(torch.abs(refined - sim_batch))
            loss_r = loss_adv + lambda_self * loss_self

            loss_r.backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), max_norm=5.0)
            opt_r.step()

            ep_loss_r += loss_adv.item()
            ep_loss_d += loss_d.item()
            ep_loss_self += loss_self.item()
            n_batches += 1

        # Epoch summary
        avg_r = ep_loss_r / n_batches
        avg_d = ep_loss_d / n_batches
        avg_s = ep_loss_self / n_batches
        elapsed = time.time() - t0
        log.append([epoch, avg_r, avg_d, avg_s])

        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch:4d}/{args.epochs}  '
                  f'L_adv={avg_r:.4f}  L_disc={avg_d:.4f}  L_self={avg_s:.4f}  '
                  f'({elapsed:.0f}s)')

        # Decay lambda_self
        if args.lambda_decay > 0 and epoch % 50 == 0:
            lambda_self = max(args.lambda_min, lambda_self * args.lambda_decay)
            print(f'  λ_self decayed to {lambda_self:.1f}')

        # Preview
        if epoch % args.preview_every == 0 or epoch == 1:
            refiner.eval()
            with torch.no_grad():
                refined_fixed = refiner(sim_fixed).cpu().numpy()
            make_preview(sim_fixed.cpu().numpy(), refined_fixed, real_fixed,
                         epoch, preview_dir, n_show=n_preview)

        # Checkpoint
        if epoch % args.ckpt_every == 0:
            torch.save({
                'epoch': epoch,
                'refiner': refiner.state_dict(),
                'disc': disc.state_dict(),
                'opt_r': opt_r.state_dict(),
                'opt_d': opt_d.state_dict(),
                'lambda_self': lambda_self,
            }, os.path.join(ckpt_dir, f'simgan_epoch_{epoch:04d}.pt'))

    # ── Save final ────────────────────────────────────────────────────────
    torch.save({
        'epoch': args.epochs,
        'refiner': refiner.state_dict(),
        'disc': disc.state_dict(),
        'lambda_self': lambda_self,
        'bands': [band_names[i] for i in band_idx],
        'n_channels': n_channels,
    }, os.path.join(args.out_dir, 'simgan_final.pt'))

    np.save(os.path.join(args.out_dir, 'train_log.npy'), np.array(log))

    # Training curves
    log = np.array(log)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(log[:, 0], log[:, 1], label='L_adv (refiner)')
    ax1.plot(log[:, 0], log[:, 2], label='L_disc')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.set_title('Adversarial losses'); ax1.grid(True, alpha=0.3)

    ax2.plot(log[:, 0], log[:, 3], color='green')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('L1 self-reg')
    ax2.set_title('Self-regularization (lower = more faithful)'); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'training_curves.png'), dpi=150)
    plt.close()

    total = time.time() - t0
    print(f'\nDone — {args.epochs} epochs in {total/60:.1f} min')
    print(f'Saved: {args.out_dir}/simgan_final.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimGAN refiner')
    parser.add_argument('--data', default='output/gan', help='Prepared data directory')
    parser.add_argument('--out_dir', default='output/v5', help='Output directory')
    parser.add_argument('--bands', default='all', help='Band(s): F115W, F150W, F277W, F444W, or all')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lambda_self', type=float, default=1000.0,
                        help='Self-regularization weight (preserves structure)')
    parser.add_argument('--lambda_decay', type=float, default=0.5,
                        help='Decay factor for lambda_self every 50 epochs')
    parser.add_argument('--lambda_min', type=float, default=100.0,
                        help='Minimum lambda_self after decay')
    parser.add_argument('--preview_every', type=int, default=10)
    parser.add_argument('--ckpt_every', type=int, default=25)
    args = parser.parse_args()
    train(args)
