"""Phase 2: Skill discovery via Wasserstein clustering and InfoNCE embedding."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split

from .data import D_ACTION


# ── Gaussian Segment Representation ────────────────────────────────────

def segment_to_gaussian(segment: torch.Tensor, eps: float = 1e-4):
    """Summarize a segment as (mean, diagonal_covariance).
    Args: segment (n, D_ACTION)
    Returns: mu (D_ACTION,), sigma_diag (D_ACTION,)
    """
    mu = segment.mean(dim=0)
    if segment.shape[0] < 2:
        sigma_diag = torch.full((D_ACTION,), eps)
    else:
        sigma_diag = segment.var(dim=0, unbiased=False) + eps
    return mu, sigma_diag


def segments_to_summaries(segments: list[torch.Tensor]) -> torch.Tensor:
    """Convert list of segments to summary matrix.
    Returns: (N, 2*D_ACTION) — concat of mu and sigma_diag per segment.
    """
    summaries = []
    for seg in segments:
        mu, sig = segment_to_gaussian(seg)
        summaries.append(torch.cat([mu, sig]))
    return torch.stack(summaries)


# ── Wasserstein-style Distance Matrix ──────────────────────────────────

def compute_distance_matrix(summaries: torch.Tensor) -> np.ndarray:
    """Compute pairwise distance: D(i,j) = ||mu_i-mu_j||² + ||sig_i-sig_j||².
    Args: summaries (N, 2*D_ACTION)
    Returns: dist_matrix (N, N) as numpy array.
    """
    N = summaries.shape[0]
    mu = summaries[:, :D_ACTION]    # (N, d)
    sig = summaries[:, D_ACTION:]   # (N, d)

    # ||mu_i - mu_j||^2
    mu_dist = torch.cdist(mu, mu, p=2).pow(2)
    # ||sig_i - sig_j||^2
    sig_dist = torch.cdist(sig, sig, p=2).pow(2)

    dist = (mu_dist + sig_dist).numpy()
    np.fill_diagonal(dist, 0.0)
    return dist


# ── Agglomerative Clustering ──────────────────────────────────────────

def cluster_segments(dist_matrix: np.ndarray, n_clusters: int = 12) -> np.ndarray:
    """Agglomerative clustering on precomputed distance matrix.
    Returns: labels (N,)
    """
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    )
    return clustering.fit_predict(dist_matrix)


# ── InfoNCE Contrastive Encoder ───────────────────────────────────────

class SegmentEncoder(nn.Module):
    """MLP encoder mapping segment summaries to latent skill space."""

    def __init__(self, d_in: int = 2 * D_ACTION, d_hidden: int = 64, d_latent: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.ReLU(),
            nn.Linear(d_hidden // 2, d_latent),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalized embeddings (B, d_latent)."""
        z = self.net(x)
        return F.normalize(z, dim=1)


def supervised_contrastive_loss(z: torch.Tensor, labels: torch.Tensor,
                                temperature: float = 0.07) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al., 2020).
    Args:
        z: (B, d) L2-normalized embeddings
        labels: (B,) integer class labels
    """
    B = z.shape[0]
    sim = z @ z.T / temperature  # (B, B)

    # Mask: same label = positive (excluding self)
    label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
    self_mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
    pos_mask = label_eq & self_mask
    neg_mask = ~label_eq & self_mask

    # For numerical stability
    sim_max = sim.max(dim=1, keepdim=True).values.detach()
    sim = sim - sim_max

    exp_sim = torch.exp(sim) * self_mask.float()
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    # Mean of log-prob over positive pairs
    pos_count = pos_mask.float().sum(dim=1).clamp(min=1)
    loss = -(log_prob * pos_mask.float()).sum(dim=1) / pos_count
    return loss.mean()


class BalancedSampler:
    """Sample balanced batches to handle class imbalance."""

    def __init__(self, labels: torch.Tensor, batch_size: int = 256):
        self.labels = labels
        self.batch_size = batch_size
        self.classes = labels.unique().tolist()
        self.class_indices = {c: (labels == c).nonzero(as_tuple=True)[0].tolist()
                              for c in self.classes}
        self.per_class = max(batch_size // len(self.classes), 2)

    def sample(self) -> torch.Tensor:
        """Return indices for one balanced batch."""
        indices = []
        for c in self.classes:
            pool = self.class_indices[c]
            n = min(self.per_class, len(pool))
            indices.extend(np.random.choice(pool, n, replace=len(pool) < n).tolist())
        np.random.shuffle(indices)
        return torch.tensor(indices[:self.batch_size])


def train_encoder(summaries: torch.Tensor, labels: torch.Tensor,
                  device: str = "cpu", epochs: int = 200,
                  lr: float = 1e-3, batch_size: int = 256,
                  verbose: bool = True) -> tuple:
    """Train InfoNCE contrastive encoder.
    Returns: (encoder, train_losses, val_losses)
    """
    # Train/val split (80/20 stratified)
    idx = np.arange(len(labels))
    train_idx, val_idx = train_test_split(idx, test_size=0.2,
                                          stratify=labels.numpy(), random_state=42)

    X_train = summaries[train_idx].to(device)
    y_train = labels[train_idx].to(device)
    X_val = summaries[val_idx].to(device)
    y_val = labels[val_idx].to(device)

    encoder = SegmentEncoder().to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)
    sampler = BalancedSampler(y_train, batch_size)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # Train
        encoder.train()
        batch_idx = sampler.sample().to(device)
        z = encoder(X_train[batch_idx])
        loss = supervised_contrastive_loss(z, y_train[batch_idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Val
        if (epoch + 1) % 10 == 0:
            encoder.eval()
            with torch.no_grad():
                z_val = encoder(X_val)
                val_loss = supervised_contrastive_loss(z_val, y_val)
            val_losses.append(val_loss.item())
            if verbose and (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: train={loss.item():.4f} val={val_loss.item():.4f}")

    return encoder, train_losses, val_losses


def encode_all(encoder: SegmentEncoder, summaries: torch.Tensor,
               device: str = "cpu") -> torch.Tensor:
    """Encode all segments to latent space. Returns (N, d_latent) on CPU."""
    encoder.eval()
    with torch.no_grad():
        z = encoder(summaries.to(device))
    return z.cpu()
