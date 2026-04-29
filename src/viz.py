# Visualisation utilities for the DDPM project.
# Extracted from ddpm.py to keep the model library free of plotting code.
#
# Author: Grégory Sainton
# Institution: Observatoire de Paris - PSL University

import numpy as np
import matplotlib.pyplot as plt
import torch


def show_images(images, suptitle="Generated images", save_path=None, show=True):
    """
    Display a batch of images in a square grid.

    Args:
        images (torch.Tensor or np.ndarray): Images (B, C, H, W) or (B, H, W, C).
        suptitle (str): Figure title.
        save_path (str, optional): File path to save the figure. Not saved if None.
        show (bool): Whether to call plt.show().
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().detach().numpy()

    # (B, C, H, W) -> (B, H, W, C)
    if images.ndim == 4 and images.shape[1] in (1, 3):
        images = np.transpose(images, (0, 2, 3, 1))

    n = len(images)
    rows = int(n ** 0.5)
    cols = int(np.ceil(n / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            img = images[i].squeeze()
            cmap = "gray" if img.ndim == 2 else None
            ax.imshow(img.clip(0, 1), cmap=cmap)

    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)


def show_forward_process(ddpm, loader, device, n_steps_to_show=4):
    """
    Visualise the forward noising process at several noise levels.

    Args:
        ddpm (DDPMModel): Trained or untrained DDPM model.
        loader (DataLoader): Data loader to draw one batch from.
        device (torch.device): Computation device.
        n_steps_to_show (int): Number of noise levels to display.
    """
    batch = next(iter(loader))
    x_0 = batch[0]
    show_images(x_0, suptitle="Original images")

    fractions = np.linspace(0, 1, n_steps_to_show + 1)[1:]
    for frac in fractions:
        t_val = int(frac * ddpm.n_steps) - 1
        t_batch = [t_val] * len(x_0)
        x_t = ddpm.q_sample(x_0.to(device), torch.tensor(t_batch).to(device))
        show_images(x_t, suptitle=f"Noisy images — {int(frac * 100)}%")


def show_noising_sequence(ddpm, loader, device, n_steps=10, t_max=None,
                          save_path=None, show=True):
    """
    Display one spiral galaxy image at n_steps noise levels, arranged horizontally.

    Picks the first image from the loader and applies the forward diffusion
    process q(x_t | x_0) at evenly spaced timesteps from 0 to t_max.

    Args:
        ddpm (DDPMModel): DDPM model (used for q_sample and schedule).
        loader (DataLoader): Data loader to draw one image from.
        device (torch.device): Computation device.
        n_steps (int): Number of noise levels to display.
        t_max (int, optional): Maximum timestep. Defaults to ddpm.n_steps - 1.
            Useful to visualise partial noising (e.g. t_max=200 out of 1000).
        save_path (str, optional): File path to save the figure. Not saved if None.
        show (bool): Whether to call plt.show().
    """
    x_0 = next(iter(loader))[0][:1].to(device)  # (1, C, H, W)

    if t_max is None:
        t_max = ddpm.n_steps - 1
    t_max = min(t_max, ddpm.n_steps - 1)

    timesteps = np.linspace(0, t_max, n_steps, dtype=int)

    fig, axes = plt.subplots(1, n_steps, figsize=(2.5 * n_steps, 3))
    for ax, t in zip(axes, timesteps):
        t_tensor = torch.tensor([t], device=device)
        x_t = ddpm.q_sample(x_0, t_tensor)
        img = x_t[0].cpu().detach().numpy().transpose(1, 2, 0).clip(0, 1)
        ax.imshow(img)
        ax.set_title(f"t={t}", fontsize=8)
        ax.axis("off")

    fig.suptitle("Forward diffusion process — q(x_t | x_0)", fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def make_denoising_gif(ddpm, loader, device, n_frames=20, gif_path="denoising.gif",
                       fps=8):
    """
    Generate a GIF showing the reverse diffusion process side by side with
    a real galaxy image.

    Each frame shows: [real image | denoising step t].
    Timesteps are evenly spaced from T-1 to 0.

    Args:
        ddpm (DDPMModel): Trained DDPM model.
        loader (DataLoader): Data loader to draw one real image from.
        device (torch.device): Computation device.
        n_frames (int): Number of frames in the GIF.
        gif_path (str): Output path for the GIF file.
        fps (int): Frames per second.
    """
    import imageio

    x_real = next(iter(loader))[0][:1].to(device)  # (1, C, H, W)

    # Collect denoising frames
    frames = []
    frame_steps = set(
        np.linspace(0, ddpm.n_steps - 1, n_frames, dtype=int).tolist()
    )

    x_t = torch.randn_like(x_real)
    with torch.no_grad():
        for t in reversed(range(ddpm.n_steps)):
            t_batch = torch.tensor([t], device=device)
            eps_theta = ddpm.predict_eps(x_t, t_batch)

            alpha_t = ddpm.alphas[t]
            alpha_bar_t = ddpm.alpha_bars[t]
            x_t = (1 / alpha_t.sqrt()) * (
                x_t - (1 - alpha_t) / (1 - alpha_bar_t).sqrt() * eps_theta
            )
            if t > 0:
                x_t = x_t + ddpm.betas[t].sqrt() * torch.randn_like(x_t)

            if t in frame_steps:
                # Normalise generated frame to [0, 1]
                gen = x_t[0].cpu().clone()
                gen -= gen.min()
                if gen.max() > 0:
                    gen /= gen.max()

                # Real image (already in [0, 1])
                real = x_real[0].cpu().clamp(0, 1)

                # Build side-by-side frame: real | separator | generated
                sep = torch.ones(3, gen.shape[1], 4)  # white vertical bar
                frame_tensor = torch.cat([real, sep, gen], dim=2)  # (C, H, 2W+4)

                # Convert to uint8 numpy (H, W, C)
                frame_np = (frame_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                frames.append(frame_np)

    # Reverse so GIF goes from noisy to clean
    frames = frames[::-1]

    # Hold the last frame longer
    frames += [frames[-1]] * (fps * 2)

    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"GIF saved to {gif_path}")
