"""Validation selection with an optional pretrained-baseline candidate."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .model import BaselineResidualCNN, ConcatenationBaselineCNN, FiLMBaselineCNN


@dataclass
class ValidationSelection:
    best_loss: float = float("inf")
    best_epoch: int = 0
    training_best_loss: float = float("inf")
    patience: int = 0

    def observe(self, epoch: int, val_loss: float) -> bool:
        # Preserve the original training patience independently of epoch 0.
        if val_loss < self.training_best_loss:
            self.training_best_loss = val_loss
            self.patience = 0
        else:
            self.patience += 1
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = epoch
            return True
        return False


def make_baseline_candidate(model: torch.nn.Module) -> torch.nn.Module:
    """Copy a fusion model and make its adapter an evaluation-time identity.

    The original training initialization is untouched. The saved state has
    the usual fusion architecture, so existing inference loaders still work.
    """
    candidate = copy.deepcopy(model)
    with torch.no_grad():
        if isinstance(candidate, ConcatenationBaselineCNN):
            layer = candidate.fusion_mlp[0]
            layer.weight.zero_()
            layer.weight[:, :candidate.embedding_dim].copy_(
                torch.eye(candidate.embedding_dim, device=layer.weight.device,
                          dtype=layer.weight.dtype)
            )
            layer.bias.zero_()
            # CNN1D.encode_drs ends in ReLU; the remaining ReLU is an identity.
        elif isinstance(candidate, BaselineResidualCNN):
            candidate.latent_proj[-1].weight.zero_()
            candidate.latent_proj[-1].bias.zero_()
        elif isinstance(candidate, FiLMBaselineCNN):
            candidate.film_conditioner[-1].weight.zero_()
            candidate.film_conditioner[-1].bias.zero_()
        else:
            raise TypeError("Epoch 0 requires a supported pretrained fusion model")
    return candidate.eval()


def check_selection_namespace(output_dir: str, enabled: bool) -> None:
    """Reject mixing completed folds produced by different selection policies."""
    root = Path(output_dir)
    config_path = root / "experiment_config.json"
    occupied = ((root / "loso_results.csv").exists()
                or any(root.glob("best_model_fold_*.pth")))
    if not occupied:
        return
    previous = json.loads(config_path.read_text()) if config_path.exists() else {}
    if bool(previous.get("include_baseline_epoch0", False)) != enabled:
        raise ValueError(
            "Epoch-0 selection policy differs from existing output; "
            "use a separate output_dir_override."
        )
