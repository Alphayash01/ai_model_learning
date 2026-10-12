import numpy as np


def wape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.abs(y - p).sum() / np.abs(y).sum())


def bias(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float((p - y).sum() / y.sum())


def pinball(y, q_hat, q: float) -> float:
    d = np.asarray(y, float) - np.asarray(q_hat, float)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def coverage(y, lo, hi) -> float:
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))
