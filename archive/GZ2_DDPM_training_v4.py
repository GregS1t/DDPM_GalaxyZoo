# Training script for the DDPM model on Galaxy Zoo 2 galaxies.
# Reads hyperparameters from param_GZ2.json.
#
# Author: Grégory Sainton
# Lab: Observatoire de Paris - PSL University

import os
import sys
import json
import datetime
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, random_split
import random
from torchvision.transforms import (Compose, RandomHorizontalFlip,
                                    RandomVerticalFlip, Resize, ToTensor)
from torchvision.transforms import functional as TF
from torch.utils.tensorboard import SummaryWriter

# AMP compatibility: torch.amp.GradScaler / autocast available from PyTorch 2.3.
# Fall back to torch.cuda.amp for older versions (e.g. 2.0.x on some clusters).
if hasattr(torch.amp, "GradScaler"):
    def make_scaler():
        return torch.amp.GradScaler()
    def make_autocast(device_type):
        return torch.amp.autocast(device_type=device_type)
else:
    def make_scaler():
        return torch.cuda.amp.GradScaler()
    def make_autocast(device_type):
        return torch.cuda.amp.autocast()

from ddpm import DDPMModel
from ema import EMA
from unet_v2 import UNetV2
from galaxy_zoo_dataset import GalaxyZooDataset, custom_collate
from gpu_utils import setup_device
from transform_custom import AsinhStretch, RandomDiscreteRotation

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
#SEED = 42
PARAMETERS_FILE = "param_GZ2.json"
TRAIN_RATIO = 0.8
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 2


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
os.makedirs("training_logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join("training_logs",
                         f"ddpm_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------
def train(ddpm, train_loader, val_loader, n_epochs, optimizer, device,
          output_model, patience=10, writer=None, loss_plot_path=None,
          warmup_epochs=10, val_freq=5, ema_decay=0.9999):
    """
    Training and validation loop for DDPMModel.


    """
    # Linear warmup for warmup_epochs, then cosine annealing for the rest.
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, n_epochs - warmup_epochs)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )
    scaler = make_scaler()
    ema = EMA(ddpm, decay=ema_decay)

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []  # list of (epoch, loss) tuples — x axis is irregular

    for epoch in tqdm(range(n_epochs), desc="Epochs"):
        # --- Training ---
        ddpm.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, leave=False,
                          desc=f"Epoch {epoch + 1}/{n_epochs}"):
            if batch is None:
                continue

            x_0 = batch[0].to(device)
            t = torch.randint(0, ddpm.n_steps, (len(x_0),)).to(device)

            optimizer.zero_grad(set_to_none=True)
            with make_autocast(device.type):
                loss = ddpm.compute_loss(x_0, t)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            ema.update()  # Update EMA weights after each optimizer step

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        train_losses.append(train_loss)
        logger.info(f"Epoch {epoch + 1}/{n_epochs} — train_loss={train_loss:.4f}")

        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)

        # --- Validation (every val_freq epochs) ---
        if (epoch + 1) % val_freq == 0:
            ema.apply()  # Use EMA weights for validation
            ddpm.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    if batch is None:
                        continue
                    x_0 = batch[0].to(device)
                    t = torch.randint(0, ddpm.n_steps, (len(x_0),)).to(device)
                    with make_autocast(device.type):
                        val_loss += ddpm.compute_loss(x_0, t).item()

            val_loss /= len(val_loader)
            ema.restore()  # Restore training weights after validation

            val_losses.append((epoch + 1, val_loss))
            logger.info(f"  Validation — val_loss={val_loss:.4f}")

            if writer is not None:
                writer.add_scalar("Loss/val", val_loss, epoch)

            # --- Checkpoint ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": ddpm.state_dict(),
                    "ema_shadow": ema.shadow,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                }, output_model)
                logger.info(f"  Best model saved (val_loss={best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered.")
                    break

    # --- Loss curves ---
    if loss_plot_path is not None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, len(train_losses) + 1), train_losses, label="Train loss")
        if val_losses:
            val_epochs, val_values = zip(*val_losses)
            ax.plot(val_epochs, val_values, "o-", label="Val loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True)
        ax.set_title("DDPM training loss")
        plt.tight_layout()
        fig.savefig(loss_plot_path, dpi=150)
        plt.close(fig)
        logger.info(f"Loss curve saved to {loss_plot_path}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    # --- Parameters ---
    if not os.path.exists(PARAMETERS_FILE):
        logger.error(f"'{PARAMETERS_FILE}' not found.")
        sys.exit(1)

    with open(PARAMETERS_FILE, "r") as f:
        p = json.load(f)

    batch_size = p["batch_size"]
    n_epochs = p["n_epochs"]
    warmup_epochs = p.get("warmup_epochs", 10)
    patience = p.get("patience", 10)
    lr = p["lr"]
    n_steps = p["n_steps"]
    min_beta = p["beta_start"]
    max_beta = p["beta_end"]
    morphology = p.get("morphology", "S")
    dataset_fraction = float(p.get("dataset_fraction", 1.0))
    output_model_name = p["output_model"]
    ref_column_file = p["catalog_file"]
    mapping_file = p["mapping_file"]
    data_dir = p["data_dir"]
    project_dir = p["project_dir"]
    output_dir = p["output_dir"]
    beta_schedule = p.get("beta_schedule", "cosine")
    ema_decay = p.get("ema_decay", 0.9999)
    asinh_stretch = p.get("asinh_stretch", False)
    asinh_scale = p.get("asinh_scale", 0.02)


    val_freq = p.get("val_freq", 5)

    SEED = p.get("seed", 42)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    logger.info(f"PyTorch {torch.__version__}")
    logger.info(f"n_epochs={n_epochs}  batch_size={batch_size}  "
                f"lr={lr}  n_steps={n_steps}")

    # --- Device ---
    device = setup_device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # --- TensorBoard ---
    os.makedirs("tensorboard_logs", exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(project_dir, "tensorboard_logs"))

    # --- Data ---
    csv_path = os.path.join(data_dir, ref_column_file)
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        logger.error(f"CSV file missing or empty: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df_mapping = pd.read_csv(os.path.join(data_dir, mapping_file))

    # Filter by morphology if specified, then merge with mapping to get file paths.
    if morphology:
        df_filtered = df[df["gz2_class"].str.startswith(morphology)]
        logger.info(f"Morphology filter: '{morphology}' — {df_filtered.shape[0]} galaxies found")
    else:
        df_filtered = df
        logger.info(f"No morphology filter — {df_filtered.shape[0]} galaxies")

    df_sub = pd.merge(df_filtered, df_mapping,
                         left_on="dr7objid", right_on="objid")

    n_total = max(1, int(len(df_sub) * dataset_fraction))
    df_sub = df_sub.sample(n=n_total, random_state=SEED)
    logger.info(f"Dataset: {n_total} galaxies ({dataset_fraction*100:.0f}% of filtered set)")

    # Images kept in [0, 1] (ToTensor only) — no Normalize.
    # Augmentations: flips and discrete 90-degree rotations only. (Chat with David)
    # Arbitrary-angle rotations are avoided because bilinear interpolation
    # degrades the PSF of astronomical images.
    
    base_transforms = [AsinhStretch(asinh_scale)] if asinh_stretch else []

    transform_train = Compose([
        Resize((64, 64)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomDiscreteRotation(),
        ToTensor(),
        *base_transforms,
    ])

    transform_val = Compose([
        Resize((64, 64)),
        ToTensor(),
        *base_transforms,
    ])

    full_dataset = GalaxyZooDataset(df_sub, data_dir, transform=None)
    train_size = int(len(full_dataset) * TRAIN_RATIO)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_dataset.dataset.transform = transform_train
    val_dataset.dataset.transform = transform_val

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            collate_fn=custom_collate)
    logger.info(f"Train: {train_size} samples / Val: {val_size} samples")

    # --- Model ---
    # For now it's a strict DDPM according to Ho et al. (2020) with a UNet noise predictor.
    ddpm = DDPMModel(UNetV2(n_steps), n_steps=n_steps, min_beta=min_beta,
                     max_beta=max_beta, device=device, image_chw=(3, 64, 64),
                     beta_schedule=beta_schedule)

    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=lr)
    output_model_path = os.path.join(project_dir, output_dir, output_model_name)
    os.makedirs(os.path.join(project_dir, output_dir), exist_ok=True)

    # --- Train ---
    loss_plot_path = os.path.join(project_dir, output_dir, "ddpm_loss_curves.png")
    train(ddpm, train_loader, val_loader, n_epochs, optimizer, device,
          output_model=output_model_path, patience=patience,
          writer=writer, loss_plot_path=loss_plot_path,
          warmup_epochs=warmup_epochs, val_freq=val_freq,
          ema_decay=ema_decay)

    writer.close()
    logger.info(f"Training complete. Model saved to {output_model_path}")