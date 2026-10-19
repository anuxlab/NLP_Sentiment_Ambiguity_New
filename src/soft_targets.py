"""Losses and distributional metrics for scarce-label sentiment experiments.

All functions accept PyTorch tensors and are small enough to test without a model
or dataset. Canonical DynaSent class order: negative, neutral, positive, mixed.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate_logits_targets(logits: torch.Tensor, targets: torch.Tensor) -> None:
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [batch, classes], got {tuple(logits.shape)}")
    if targets.ndim != 2 or targets.shape != logits.shape:
        raise ValueError("soft targets must have the same [batch, classes] shape as logits")
    if not torch.is_floating_point(targets):
        raise TypeError("soft targets must be floating-point probabilities")
    if not torch.isfinite(targets).all():
        raise ValueError("soft targets must be finite")
    if (targets < 0).any():
        raise ValueError("soft targets cannot contain negative values")
    sums = targets.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5):
        raise ValueError("each soft-target row must sum to 1")


def soft_target_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross entropy against human vote distributions, `-sum(y * log_softmax(z))`."""
    _validate_logits_targets(logits, targets)
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    targets = targets.to(device=logits.device, dtype=logits.dtype)
    losses = -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()


def hard_label_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Hard-label CE baseline; set smoothing=0 for majority CE, >0 for smoothing baseline."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must have shape [batch]")
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError("hard labels must be integer class indices")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    return F.cross_entropy(logits, labels.long(), reduction=reduction, label_smoothing=label_smoothing)


def normalize_probabilities(probabilities: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Clamp tiny negatives to zero and normalize each row; reject zero-mass rows."""
    if probabilities.ndim < 1:
        raise ValueError("probabilities must have at least one dimension")
    if not torch.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if (probabilities < -eps).any():
        raise ValueError("probabilities cannot contain negative values")
    probs = probabilities.clamp_min(0)
    mass = probs.sum(dim=-1, keepdim=True)
    if (mass <= eps).any():
        raise ValueError("probability rows must have positive mass")
    return probs / mass


def jensen_shannon_divergence(
    predicted_probs: torch.Tensor,
    human_probs: torch.Tensor,
    *,
    reduction: str = "mean",
    eps: float = 1e-8,
    base: float = 2.0,
) -> torch.Tensor:
    """Per-row Jensen-Shannon divergence between predicted and human distributions.

    With base=2, values are in [0, 1] bits. Supports `none`, `mean`, `sum`.
    Predictions and targets are normalized internally; gradients flow to predictions.
    """
    if predicted_probs.shape != human_probs.shape or predicted_probs.ndim != 2:
        raise ValueError("predicted_probs and human_probs must share [batch, classes] shape")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if base <= 0 or base == 1:
        raise ValueError("base must be positive and not equal to 1")
    p = normalize_probabilities(predicted_probs)
    q = normalize_probabilities(human_probs.to(device=p.device, dtype=p.dtype))
    m = 0.5 * (p + q)
    log_base = torch.log(torch.tensor(base, dtype=p.dtype, device=p.device))

    def kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # 0 * log(0/b) is defined as zero; clamp only inside logarithm.
        terms = torch.where(a > 0, a * (torch.log(a.clamp_min(eps)) - torch.log(b.clamp_min(eps))), 0.0)
        return terms.sum(dim=-1) / log_base

    values = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    return values.mean()
