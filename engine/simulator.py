import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def simulate_users(variant: Dict[str, Any], n: int, seed: int = None) -> np.ndarray:
    if n <= 0:
        raise ValueError("Number of users must be positive")
    if seed is not None:
        np.random.seed(seed)
    base_rate = 0.08
    if variant.get("cta_color") == "green":
        base_rate += 0.015
    if variant.get("discount", 0) >= 10:
        base_rate += 0.02
    base_rate = max(0.0, min(1.0, base_rate))
    logger.info(f"Simulating {n} users with conversion rate: {base_rate:.4f}")
    return np.random.binomial(1, base_rate, size=n)
