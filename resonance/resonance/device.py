"""Device detection and reproducibility utilities."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Return the best available torch device.

    Priority: preferred > CUDA > MPS > CPU.
    """
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS doesn't have a manual_seed_all, but torch.manual_seed covers it
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def enable_deterministic(mode: bool = True) -> None:
    """Enable deterministic behaviour where possible.

    Note: on MPS some operations are not deterministic; this sets the
    global flags and warns when a non-deterministic op is encountered.
    """
    if mode:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
