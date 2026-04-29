# Evaluation metrics for generative models.
#
# Implements:
#   - InceptionV3 feature extraction (shared between all metrics)
#   - Improved Precision and Recall (Kynkäänniemi et al., NeurIPS 2019)
#     https://arxiv.org/abs/1904.06991
#   - Density and Coverage (Naeem et al., ICML 2020)
#     https://arxiv.org/abs/2002.09797

# Author: Grégory Sainton
# Institution: Observatoire de Paris - PSL University

import torch


def build_inception(device):
    """
    Load InceptionV3 feature extractor from pytorch-fid.

    Returns the model in eval mode on the given device.

    Args:
        device (torch.device): Computation device.

    Returns:
        nn.Module: InceptionV3 feature extractor (output dim: 2048).
    """
    try:
        from pytorch_fid.inception import InceptionV3
    except ImportError:
        raise ImportError("pytorch-fid not installed. Run: pip install pytorch-fid")

    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]])
    inception.to(device)
    inception.eval()
    return inception


def extract_features(loader, inception, device, max_samples=1000):
    """
    Extract InceptionV3 features from a DataLoader.

    Corresponds to F(X) in Algorithm 1 of Kynkäänniemi et al. (2019).

    Args:
        loader (DataLoader): Image data loader (images in [0, 1]).
        inception (nn.Module): InceptionV3 feature extractor from build_inception().
        device (torch.device): Computation device.
        max_samples (int): Maximum number of images to process.

    Returns:
        torch.Tensor: Feature matrix of shape (N, 2048).
    """
    features = []
    n_extracted = 0

    for batch in loader:
        if batch is None:
            continue
        images = batch[0].to(device)

        with torch.no_grad():
            feat = inception(images)[0]            # (B, 2048, 1, 1)
            feat = feat.squeeze(-1).squeeze(-1)    # (B, 2048)

        features.append(feat.cpu())
        n_extracted += len(feat)

        if n_extracted >= max_samples:
            break

    return torch.cat(features, dim=0)[:max_samples]  # (N, 2048)


def pairwise_distances(a, b):
    """
    Compute pairwise Euclidean distances between two feature matrices.

    Uses the identity ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b^T.
    Corresponds to line 10 of Algorithm 1.

    Args:
        a (torch.Tensor): Feature matrix (N, D).
        b (torch.Tensor): Feature matrix (M, D).

    Returns:
        torch.Tensor: Distance matrix (N, M).
    """
    a_sq = (a ** 2).sum(dim=1, keepdim=True)   # (N, 1)
    b_sq = (b ** 2).sum(dim=1, keepdim=True)   # (M, 1)
    ab = a @ b.T                                # (N, M)
    distances = (a_sq + b_sq.T - 2 * ab).clamp(min=0).sqrt()
    return distances


def manifold_estimate(phi_a, phi_b, k):
    """
    Estimate the manifold of phi_a and count how many points from phi_b lie within it.

    Corresponds to MANIFOLD-ESTIMATE(Phi_a, Phi_b, k) in Algorithm 1:
      - For each phi in phi_a, compute pairwise distances to all points in phi_a
        and tabulate the distance to the k-th nearest neighbor (lines 9-11).
      - Count the fraction of points from phi_b that lie within the manifold (lines 13-17).

    Args:
        phi_a (torch.Tensor): Reference feature matrix (N, D). Defines the manifold.
        phi_b (torch.Tensor): Query feature matrix (M, D). Points to test membership.
        k (int): Neighborhood size — radius of each hypersphere.

    Returns:
        float: Fraction of points from phi_b within the manifold of phi_a.
    """
    phi_a = phi_a.float()
    phi_b = phi_b.float()

    # Lines 9-11: pairwise distances within phi_a, radius = k-th NN distance.
    # sorted[:, 0] = 0 (self-distance), sorted[:, k] = k-th neighbor distance.
    dist_aa = pairwise_distances(phi_a, phi_a)          # (N, N)
    sorted_aa, _ = dist_aa.sort(dim=1)
    radii = sorted_aa[:, k]                             # (N,) — r_phi for each phi in phi_a

    # Lines 13-17: for each phi in phi_b, check if it lies within any hypersphere of phi_a.
    dist_ba = pairwise_distances(phi_b, phi_a)          # (M, N)
    in_manifold = (dist_ba <= radii.unsqueeze(0)).any(dim=1)  # (M,)

    return in_manifold.float().mean().item()


def precision_recall(phi_r, phi_g, k=3):
    """
    Compute Improved Precision and Recall (Kynkäänniemi et al., NeurIPS 2019).

    Corresponds to PRECISION-RECALL(Xr, Xg, F, k) in Algorithm 1:
      precision = MANIFOLD-ESTIMATE(Phi_r, Phi_g, k)
      recall    = MANIFOLD-ESTIMATE(Phi_g, Phi_r, k)

    Args:
        phi_r (torch.Tensor): Real image features (N, D).
        phi_g (torch.Tensor): Generated image features (M, D).
        k (int): Neighborhood size. Default: 3 (recommended by the authors).

    Returns:
        tuple: (precision, recall) as float values in [0, 1].
    """
    precision = manifold_estimate(phi_r, phi_g, k)
    recall = manifold_estimate(phi_g, phi_r, k)
    return precision, recall


def density_coverage(phi_r, phi_g, k=5):
    """
    Compute Density and Coverage metrics (Naeem et al., ICML 2020).

    Density improves upon Precision by counting how many real hyperspheres
    contain each generated sample (weighted count rather than binary).
    Coverage improves upon Recall by building hyperspheres around real samples
    rather than generated ones, reducing sensitivity to generated outliers.

    Density = (1 / (k * M)) * sum_j sum_i 1[phi_g[j] in B(phi_r[i], r_i)]
    Coverage = (1 / N) * sum_i 1[B(phi_r[i], r_i) contains at least one phi_g]

    Args:
        phi_r (torch.Tensor): Real image features (N, D).
        phi_g (torch.Tensor): Generated image features (M, D).
        k (int): Neighborhood size for real manifold. Default: 5.

    Returns:
        tuple: (density, coverage) as floats.
            density in [0, +inf) — values > 1 indicate generated samples
            cluster in high-density real regions.
            coverage in [0, 1].
    """
    phi_r = phi_r.float()
    phi_g = phi_g.float()

    M = phi_g.shape[0]

    # Build hyperspheres around real samples — radius = k-th NN distance within phi_r
    dist_rr = pairwise_distances(phi_r, phi_r)          # (N, N)
    sorted_rr, _ = dist_rr.sort(dim=1)
    radii = sorted_rr[:, k]                             # (N,)

    # For each generated sample, count how many real hyperspheres contain it
    dist_gr = pairwise_distances(phi_g, phi_r)          # (M, N)
    inside = (dist_gr <= radii.unsqueeze(0))            # (M, N) bool

    density = inside.float().sum().item() / (k * M)
    coverage = inside.any(dim=0).float().mean().item()

    return density, coverage


def compute_fid(phi_r, phi_g):
    """
    Compute FID from pre-extracted InceptionV3 features.

    FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2*(Sigma_r @ Sigma_g)^0.5)

    Args:
        phi_r (torch.Tensor): Real features (N, 2048).
        phi_g (torch.Tensor): Generated features (M, 2048).

    Returns:
        float: FID score. Lower is better.
    """
    import numpy as np
    from scipy.linalg import sqrtm

    mu_r = phi_r.mean(0).numpy()
    mu_g = phi_g.mean(0).numpy()
    sigma_r = np.cov(phi_r.numpy(), rowvar=False)
    sigma_g = np.cov(phi_g.numpy(), rowvar=False)

    diff = mu_r - mu_g
    covmean = sqrtm(sigma_r @ sigma_g)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)
    return float(fid)
