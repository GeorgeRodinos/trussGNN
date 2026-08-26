"""Configuration for deterministic model training."""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class TrainingConfig:
    """Small set of options needed by the Phase 4D2 training loop."""

    max_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 10
    min_delta: float = 0.0
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not isinstance(self.max_epochs, int):
            raise ValueError("max_epochs must be an integer")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be at least 1")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if not isinstance(self.patience, int):
            raise ValueError("patience must be an integer")
        if self.patience < 0:
            raise ValueError("patience cannot be negative")
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("min_delta cannot be negative")
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        try:
            device = torch.device(self.device)
        except (RuntimeError, ValueError) as error:
            raise ValueError(f"Invalid device: {self.device}") from error
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
