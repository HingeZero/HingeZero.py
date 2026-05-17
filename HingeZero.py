import numpy as np

def hingezero_xy(x, y, alpha=2.0, z0=0.0):
    """
    HingeZero dual-tanh coupling.

    x, y  : input vectors or scalars
    alpha : hinge sharpness / gain
    z0    : optional zero-region offset
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    hx = np.tanh(alpha * (x - z0))
    hy = np.tanh(alpha * (y + z0))

    return hx * hy


def hingezero_update(x, y, steps=2, eta=0.1, lam=0.02, alpha=2.0, z0=0.0):
    """
    Iterative HingeZero stabilisation.
    """
    x = np.asarray(x, dtype=np.float32)

    for _ in range(steps):
        h = hingezero_xy(x, y, alpha=alpha, z0=z0)
        x = x + eta * h - lam * x
        x = x / (np.linalg.norm(x) + 1e-8)

    return x
