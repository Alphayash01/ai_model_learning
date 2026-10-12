"""Exercise 4: forecast metrics. Work two of them out by hand on paper first.

Run:  pytest -k ex04 -q
"""
import numpy as np


def wape(y, p) -> float:
    """Weighted absolute percentage error: sum(|y - p|) / sum(|y|)."""
    raise NotImplementedError


def bias(y, p) -> float:
    """Signed error: sum(p - y) / sum(y). Positive means over-forecast."""
    raise NotImplementedError


def pinball(y, q_hat, q: float) -> float:
    """Pinball (quantile) loss, averaged.

    d = y - q_hat ; loss = max(q * d, (q - 1) * d)
    """
    raise NotImplementedError


def coverage(y, lo, hi) -> float:
    """Share of actuals that fall inside [lo, hi] (inclusive)."""
    raise NotImplementedError
