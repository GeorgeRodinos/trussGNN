"""Configuration for deterministic model training."""

import math
from dataclasses import dataclass
from numbers import Real

import torch


@dataclass(frozen=True)
class TrainingConfig:
    """Small set of options needed by the deterministic training loop."""

    max_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 10
    min_delta: float = 0.0
    seed: int = 42
    device: str = "cpu"
    physics_loss_weight: float = 0.0
    physics_epsilon: float = 1e-12

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
        if isinstance(self.physics_loss_weight, bool) or not isinstance(
            self.physics_loss_weight, Real
        ):
            raise ValueError("physics_loss_weight must be finite and non-negative")
        if (
            not math.isfinite(self.physics_loss_weight)
            or self.physics_loss_weight < 0
        ):
            raise ValueError("physics_loss_weight must be finite and non-negative")
        if isinstance(self.physics_epsilon, bool) or not isinstance(
            self.physics_epsilon, Real
        ):
            raise ValueError("physics_epsilon must be finite and positive")
        if not math.isfinite(self.physics_epsilon) or self.physics_epsilon <= 0:
            raise ValueError("physics_epsilon must be finite and positive")
        try:
            device = torch.device(self.device)
        except (RuntimeError, ValueError) as error:
            raise ValueError(f"Invalid device: {self.device}") from error
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
