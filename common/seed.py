"""Global determinism utility.

Every script that generates data, trains a model or scores an event calls
:func:`set_global_seed` first. That single call seeds Python's ``random``, NumPy, and
PyTorch (if installed), and pins ``PYTHONHASHSEED`` so hash-ordered iteration is stable
across processes.

LightGBM is configured separately via :func:`lightgbm_params`, because it needs
determinism flags (single-threaded, ``deterministic=True``) rather than a global seed.

Example
-------
>>> from common.seed import set_global_seed
>>> set_global_seed()
42
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional

from common.config import settings


def set_global_seed(seed: Optional[int] = None) -> int:
    """Seed every source of randomness in the process.

    Parameters
    ----------
    seed:
        Seed to apply. Defaults to ``settings.random_seed`` (42).

    Returns
    -------
    int
        The seed that was actually applied, so callers can log it.
    """
    resolved = settings.random_seed if seed is None else int(seed)

    # Stable hashing for dict/set iteration in child processes.
    os.environ["PYTHONHASHSEED"] = str(resolved)

    random.seed(resolved)

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard dependency in practice
        pass
    else:
        np.random.seed(resolved)

    try:  # PyTorch is optional at import time (light test installs skip it).
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(resolved)
        torch.cuda.manual_seed_all(resolved)
        # Single-threaded CPU math keeps float reductions bit-for-bit reproducible.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_num_threads(1)

    return resolved


def new_rng(seed: Optional[int] = None) -> "Any":
    """Return an isolated NumPy ``Generator``.

    Preferred over the global NumPy state when a component needs its own reproducible
    stream that cannot be disturbed by unrelated code.
    """
    import numpy as np

    resolved = settings.random_seed if seed is None else int(seed)
    return np.random.default_rng(resolved)


def lightgbm_params(seed: Optional[int] = None, **overrides: Any) -> Dict[str, Any]:
    """Return LightGBM parameters that make training bit-for-bit reproducible.

    Determinism in LightGBM requires more than a seed: feature/bagging/data seeds must
    all be fixed, threading must be single, and ``deterministic`` must be enabled.

    Parameters
    ----------
    seed:
        Base seed. Defaults to ``settings.random_seed``.
    **overrides:
        Extra parameters merged on top (e.g. ``objective``, ``num_class``).
    """
    resolved = settings.random_seed if seed is None else int(seed)
    params: Dict[str, Any] = {
        "seed": resolved,
        "random_state": resolved,
        "bagging_seed": resolved,
        "feature_fraction_seed": resolved,
        "data_random_seed": resolved,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "verbose": -1,
    }
    params.update(overrides)
    return params


__all__ = ["set_global_seed", "new_rng", "lightgbm_params"]
