"""Phase 3: Hierarchical skill composition via sequence model."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import N_SKILLS, SKILL_TO_IDX, SKILL_TYPES


# ── Transformer-based Macro Policy ───────────────────────────────────

class TransformerMacroPolicy(nn.Module):
    """Transformer decoder that predicts the next skill given previous skills.
    Input: sequence of skill embeddings (B, T, d_latent)
    Output: next-skill logits (B, T, n_skills), termination probs (B, T, 1)
    """

    def __init__(self, d_latent: int = 16, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2,
                 n_skills: int = N_SKILLS, max_len: int = 20,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Project skill embeddings to model dimension
        self.input_proj = nn.Linear(d_latent, d_model)

        # Learned positional encoding
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Transformer decoder layers (causal self-attention)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output heads
        self.skill_head = nn.Linear(d_model, n_skills)
        self.term_head = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, z_seq: torch.Tensor) -> tuple:
        """
        Args: z_seq (B, T, d_latent) — sequence of skill embeddings
        Returns: (skill_logits (B, T, n_skills), term_prob (B, T, 1))
        """
        B, T, _ = z_seq.shape
        device = z_seq.device

        # Project and add positional encoding
        h = self.input_proj(z_seq)  # (B, T, d_model)
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        h = h + self.pos_emb(positions)

        # Causal mask: each position can only attend to itself and earlier
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)

        # Transformer (self-attention only, no cross-attention memory needed)
        # Use dummy memory of zeros
        memory = torch.zeros(B, 1, self.d_model, device=device)
        h = self.transformer(h, memory, tgt_mask=causal_mask)

        skill_logits = self.skill_head(h)  # (B, T, n_skills)
        term_prob = torch.sigmoid(self.term_head(h))  # (B, T, 1)
        return skill_logits, term_prob


# ── Legacy MLP Policy (kept for comparison) ──────────────────────────

class MacroPolicy(nn.Module):
    """Simple MLP baseline: predicts next skill from (z_current, position)."""

    def __init__(self, d_latent: int = 16, d_hidden: int = 64, n_skills: int = N_SKILLS):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(d_latent + 1, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.ReLU(),
        )
        self.skill_head = nn.Linear(d_hidden // 2, n_skills)
        self.term_head = nn.Linear(d_hidden // 2, 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.skill_head(h), torch.sigmoid(self.term_head(h))


# ── Dataset Building ─────────────────────────────────────────────────

def build_sequence_dataset(traj_data_list, encoder, summaries, label_ints,
                           device="cpu"):
    """Build variable-length sequence dataset for Transformer training.
    Each trajectory becomes a sequence of (skill_embedding, skill_label) pairs.
    Returns: list of (z_seq Tensor (T, d_latent), label_seq Tensor (T,))
    """
    encoder.eval()
    with torch.no_grad():
        z_all = encoder(summaries.to(device)).cpu()

    sequences = []
    seg_idx = 0

    for td in traj_data_list:
        n_segs = len(td.gt_segments)
        if n_segs < 2:
            seg_idx += n_segs
            continue

        z_seq = z_all[seg_idx:seg_idx + n_segs]  # (n_segs, d_latent)
        label_seq = torch.tensor([SKILL_TO_IDX[l] for l in td.gt_segment_labels])

        sequences.append((z_seq, label_seq))
        seg_idx += n_segs

    return sequences


def build_composition_dataset(traj_data_list, encoder, summaries, label_ints,
                              device="cpu"):
    """Build training data for MLP macro policy (legacy).
    Returns: states (M, 17), skill_targets (M,), term_targets (M,)
    """
    encoder.eval()
    with torch.no_grad():
        z_all = encoder(summaries.to(device)).cpu()

    states, skill_targets, term_targets = [], [], []
    seg_idx = 0

    for td in traj_data_list:
        n_segs = len(td.gt_segments)
        if n_segs < 2:
            seg_idx += n_segs
            continue

        for k in range(n_segs - 1):
            z_k = z_all[seg_idx + k]
            pos = torch.tensor([(k + 1) / n_segs])
            state = torch.cat([z_k, pos])
            next_label = SKILL_TO_IDX[td.gt_segment_labels[k + 1]]

            states.append(state)
            skill_targets.append(next_label)
            term_targets.append(1.0)

            pos_mid = torch.tensor([k / n_segs + 0.01])
            state_mid = torch.cat([z_k, pos_mid])
            states.append(state_mid)
            skill_targets.append(SKILL_TO_IDX[td.gt_segment_labels[k]])
            term_targets.append(0.0)

        seg_idx += n_segs

    return (torch.stack(states),
            torch.tensor(skill_targets, dtype=torch.long),
            torch.tensor(term_targets, dtype=torch.float))


# ── Transformer Training ─────────────────────────────────────────────

def collate_sequences(sequences, max_len=None):
    """Pad variable-length sequences into a batch.
    Returns: z_batch (B, T, d), label_batch (B, T), mask (B, T)
    """
    if max_len is None:
        max_len = max(s[0].shape[0] for s in sequences)

    B = len(sequences)
    d = sequences[0][0].shape[1]
    z_batch = torch.zeros(B, max_len, d)
    label_batch = torch.zeros(B, max_len, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i, (z_seq, label_seq) in enumerate(sequences):
        T = min(z_seq.shape[0], max_len)
        z_batch[i, :T] = z_seq[:T]
        label_batch[i, :T] = label_seq[:T]
        mask[i, :T] = True

    return z_batch, label_batch, mask


def train_transformer_policy(sequences, device="cpu", epochs=300,
                             lr=5e-4, batch_size=64, verbose=True):
    """Train the Transformer macro policy on sequence data.
    The model predicts skill[t+1] from skills[0:t+1] (teacher forcing).
    Returns: (policy, train_losses, val_losses)
    """
    # Train/val split
    np.random.seed(42)
    idx = np.random.permutation(len(sequences))
    split = int(0.85 * len(idx))
    train_seqs = [sequences[i] for i in idx[:split]]
    val_seqs = [sequences[i] for i in idx[split:]]

    policy = TransformerMacroPolicy().to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        policy.train()

        # Sample a mini-batch
        batch_idx = np.random.choice(len(train_seqs), min(batch_size, len(train_seqs)),
                                     replace=False)
        batch = [train_seqs[i] for i in batch_idx]
        z_batch, label_batch, mask = collate_sequences(batch)
        z_batch = z_batch.to(device)
        label_batch = label_batch.to(device)
        mask = mask.to(device)

        # Teacher forcing: input is skills[0:T-1], target is skills[1:T]
        z_input = z_batch[:, :-1]    # (B, T-1, d)
        targets = label_batch[:, 1:]  # (B, T-1)
        target_mask = mask[:, 1:]     # (B, T-1)

        if z_input.shape[1] == 0:
            continue

        skill_logits, term_prob = policy(z_input)  # (B, T-1, n_skills)

        # Masked cross-entropy loss
        logits_flat = skill_logits[target_mask]   # (M, n_skills)
        targets_flat = targets[target_mask]       # (M,)

        if logits_flat.shape[0] == 0:
            continue

        loss = F.cross_entropy(logits_flat, targets_flat)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        train_losses.append(loss.item())

        # Validation
        if (epoch + 1) % 10 == 0:
            policy.eval()
            with torch.no_grad():
                z_val, lbl_val, mask_val = collate_sequences(val_seqs)
                z_val = z_val.to(device)
                lbl_val = lbl_val.to(device)
                mask_val = mask_val.to(device)

                z_in = z_val[:, :-1]
                tgt = lbl_val[:, 1:]
                tgt_m = mask_val[:, 1:]

                if z_in.shape[1] > 0:
                    sl, _ = policy(z_in)
                    vl = F.cross_entropy(sl[tgt_m], tgt[tgt_m])
                    val_losses.append(vl.item())

                    if verbose and (epoch + 1) % 50 == 0:
                        acc = (sl[tgt_m].argmax(dim=1) == tgt[tgt_m]).float().mean()
                        print(f"  Epoch {epoch+1}/{epochs}: "
                              f"train={loss.item():.4f} val={vl.item():.4f} "
                              f"acc={acc.item():.3f}")

    return policy, train_losses, val_losses


# ── Legacy MLP Training ──────────────────────────────────────────────

def train_macro_policy(states, skill_targets, term_targets,
                       device="cpu", epochs=100, lr=1e-3, batch_size=128,
                       verbose=True):
    """Train the MLP macro policy (legacy). Returns (policy, train_losses)."""
    N = states.shape[0]
    states = states.to(device)
    skill_targets = skill_targets.to(device)
    term_targets = term_targets.to(device)

    policy = MacroPolicy().to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    losses = []

    for epoch in range(epochs):
        policy.train()
        idx = torch.randperm(N, device=device)[:batch_size]

        skill_logits, term_prob = policy(states[idx])
        loss_skill = F.cross_entropy(skill_logits, skill_targets[idx])
        loss_term = F.binary_cross_entropy(term_prob.squeeze(), term_targets[idx])
        loss = loss_skill + loss_term

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if verbose and (epoch + 1) % 25 == 0:
            acc = (skill_logits.argmax(dim=1) == skill_targets[idx]).float().mean()
            print(f"  Epoch {epoch+1}/{epochs}: loss={loss.item():.4f} "
                  f"skill_acc={acc.item():.3f}")

    return policy, losses


# ── Sequence Reconstruction ──────────────────────────────────────────

def reconstruct_sequences_transformer(policy, encoder, traj_data_list,
                                      summaries, device="cpu", max_len=10):
    """Autoregressively reconstruct skill sequences using Transformer policy.
    Uses the actual segment embeddings for the first step, then feeds back
    the prototype of each predicted skill.
    """
    encoder.eval()
    policy.eval()

    with torch.no_grad():
        z_all = encoder(summaries.to(device)).cpu()

    # Build prototypes
    label_ints = []
    for td in traj_data_list:
        for lbl in td.gt_segment_labels:
            label_ints.append(SKILL_TO_IDX[lbl])
    label_ints_t = torch.tensor(label_ints)

    prototypes = {}
    for i, name in enumerate(SKILL_TYPES):
        mask = label_ints_t == i
        if mask.any():
            prototypes[i] = z_all[mask].mean(dim=0)

    pred_sequences = []
    seg_idx = 0

    for td in traj_data_list:
        n_segs = len(td.gt_segments)
        if n_segs < 1:
            pred_sequences.append([])
            continue

        # Start with first segment's actual embedding
        z_history = [z_all[seg_idx].unsqueeze(0)]  # list of (1, d_latent)
        pred_seq = [td.gt_segment_labels[0]]  # first skill given

        for step in range(1, min(n_segs, max_len)):
            # Build input sequence from history
            z_input = torch.stack([z.squeeze(0) for z in z_history]).unsqueeze(0)  # (1, t, d)
            z_input = z_input.to(device)

            with torch.no_grad():
                skill_logits, _ = policy(z_input)  # (1, t, n_skills)

            # Take prediction from last position
            pred_skill_idx = skill_logits[0, -1].argmax().item()
            pred_seq.append(SKILL_TYPES[pred_skill_idx])

            # Use prototype for next step's input
            if pred_skill_idx in prototypes:
                z_next = prototypes[pred_skill_idx]
            else:
                z_next = z_history[-1].squeeze(0)
            z_history.append(z_next.unsqueeze(0))

        pred_sequences.append(pred_seq)
        seg_idx += n_segs

    return pred_sequences


def reconstruct_sequences(policy, encoder, traj_data_list, summaries,
                          device="cpu", max_len=10):
    """Legacy MLP-based reconstruction."""
    encoder.eval()
    policy.eval()

    with torch.no_grad():
        z_all = encoder(summaries.to(device)).cpu()

    label_ints = []
    for td in traj_data_list:
        for lbl in td.gt_segment_labels:
            label_ints.append(SKILL_TO_IDX[lbl])
    label_ints_t = torch.tensor(label_ints)

    prototypes = {}
    for i, name in enumerate(SKILL_TYPES):
        mask = label_ints_t == i
        if mask.any():
            prototypes[i] = z_all[mask].mean(dim=0)

    pred_sequences = []
    seg_idx = 0

    for td in traj_data_list:
        n_segs = len(td.gt_segments)
        if n_segs < 1:
            pred_sequences.append([])
            continue

        z_cur = z_all[seg_idx]
        pred_seq = [td.gt_segment_labels[0]]

        for step in range(1, min(n_segs, max_len)):
            pos = torch.tensor([step / n_segs])
            state = torch.cat([z_cur, pos]).unsqueeze(0).to(device)

            with torch.no_grad():
                skill_logits, term_prob = policy(state)

            pred_skill_idx = skill_logits.argmax(dim=1).item()
            pred_seq.append(SKILL_TYPES[pred_skill_idx])

            if pred_skill_idx in prototypes:
                z_cur = prototypes[pred_skill_idx]

        pred_sequences.append(pred_seq)
        seg_idx += n_segs

    return pred_sequences
