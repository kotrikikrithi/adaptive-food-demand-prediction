"""
A small neural network (MLP) and a training loop built for stability.

Stability tools used:
  - Huber loss            : robust to outlier days (festivals, event spikes)
  - AdamW weight decay    : L2 regularisation against overfitting
  - Dropout               : more regularisation
  - Gradient clipping     : gradients can never explode (max norm = config.GRAD_CLIP_NORM)
  - Early stopping        : stop when validation loss stops improving
  - Fixed seeds + deterministic algorithms : same input -> same result, every run
"""
import copy
import random
import time

import numpy as np
import torch
from torch import nn

import config

ACT = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class MLP(nn.Module):
    def __init__(self, in_dim, n_layers, width, dropout, activation):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, width), ACT[activation](), nn.Dropout(dropout)]
            d = width
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(X_tr, y_tr, X_va, y_va, hp, seed=config.SEED, max_epochs=config.MAX_EPOCHS,
                patience=config.PATIENCE, init_model=None, lr=None, sample_weight=None):
    """Train (or fine-tune, if init_model is given). Returns (model, history, diverged).

    sample_weight (optional): per-row importance, e.g. to make recent days count more.
    """
    set_seed(seed)
    X_tr, y_tr = torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.float32)
    has_val = X_va is not None and len(X_va) > 0
    if has_val:
        X_va, y_va = torch.tensor(X_va), torch.tensor(y_va, dtype=torch.float32)

    model = copy.deepcopy(init_model) if init_model is not None else \
        MLP(X_tr.shape[1], hp["n_layers"], hp["width"], hp["dropout"], hp["activation"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr or hp["lr"], weight_decay=hp["weight_decay"])
    loss_fn = nn.HuberLoss(delta=hp["huber_delta"])
    w = None if sample_weight is None else torch.tensor(sample_weight, dtype=torch.float32)
    weighted_fn = nn.HuberLoss(delta=hp["huber_delta"], reduction="none")
    gen = torch.Generator().manual_seed(seed)

    history, best_state, best_val, bad = [], None, float("inf"), 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(X_tr), generator=gen)
        total, max_gn = 0.0, 0.0
        for i in range(0, len(X_tr), config.BATCH_SIZE):
            idx = perm[i:i + config.BATCH_SIZE]
            opt.zero_grad()
            if w is None:
                loss = loss_fn(model(X_tr[idx]), y_tr[idx])
            else:
                loss = (weighted_fn(model(X_tr[idx]), y_tr[idx]) * w[idx]).mean()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            max_gn = max(max_gn, float(gn))
            opt.step()
            total += loss.item() * len(idx)
        train_loss = total / len(X_tr)
        if not np.isfinite(train_loss):
            return model, history, True  # diverged (should never happen thanks to clipping)

        val_loss = float("nan")
        if has_val:
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(X_va), y_va))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "max_grad_norm_before_clip": max_gn})

        if has_val and patience:
            if val_loss < best_val - 1e-6:
                best_val, best_state, bad = val_loss, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history, False


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X)).numpy()


def measure_latency_ms(model, in_dim, n=300):
    """Average time for ONE prediction (batch of 1), in milliseconds."""
    x = torch.zeros(1, in_dim)
    model.eval()
    with torch.no_grad():
        for _ in range(20):  # warm-up
            model(x)
        t0 = time.perf_counter()
        for _ in range(n):
            model(x)
    return (time.perf_counter() - t0) / n * 1000
