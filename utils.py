"""Utility helpers: reproducible seeding, device selection, config IO."""
from __future__ import annotations
import os
import random
import json
import logging
from typing import Any, Dict

import numpy as np
import yaml

logger = logging.getLogger("herl_aid")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def set_seed(seed: int = 42) -> None:
    """Fix all RNGs for deterministic reproduction."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(prefer: str = "cuda") -> str:
    """Return an available device string, falling back to CPU."""
    try:
        import torch
        if prefer == "cuda" and torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    logger.info("wrote %s", path)
