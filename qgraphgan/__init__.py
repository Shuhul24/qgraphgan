"""QGraphGAN: a quantum graph-softmax generator for adversarial link prediction."""

import torch

# All models, circuits and metrics run in float64.
torch.set_default_dtype(torch.float64)

from .config import Config  # noqa: E402
from .train import run_experiment  # noqa: E402

__all__ = ["Config", "run_experiment"]
