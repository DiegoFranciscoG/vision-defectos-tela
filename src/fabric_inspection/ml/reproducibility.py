"""Seeds and deterministic settings (docs/investigacion.md #25)."""

import os
import random

import numpy as np
import torch

REFERENCE_THREADS = 4


def set_seed(seed: int, *, threads: int = REFERENCE_THREADS) -> torch.Generator:
    """Seed every RNG, force deterministic kernels and fix the CPU thread count.

    PyTorch does not guarantee identical results across releases, platforms or CPU/GPU, so the
    reference metrics are produced on CPU with the versions pinned in uv.lock.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(threads)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
