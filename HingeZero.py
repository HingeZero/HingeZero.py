import numpy as np

# ============================================================
# EXACT HingeZero Validation Core
# ============================================================

def hinge_phi(h, alpha=0.25):
    """
    φ(h)_i = tanh(h_i) + α * tanh(2 * h_i)
    """
    return np.tanh(h) + alpha * np.tanh(2.0 * h)


def hz_recall(
    W,
    x0,
    steps=320,
    alpha=0.25,
    eps=0.10,
    lam=0.02
):
    """
    HingeZero recall dynamics

      h_t      = W x_t
      φ(h_t)_i = tanh(h_t_i) + α·tanh(2·h_t_i)
      x_{t+1}  = (1 − λ)·x_t + ε·φ(h_t)
    """

    x = x0.copy().astype(float)

    for _ in range(steps):

        h = W @ x

        phi = hinge_phi(
            h,
            alpha=alpha
        )

        x = (
            (1.0 - lam) * x
            + eps * phi
        )

    return x


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":

    rng = np.random.default_rng(0)

    N = 256
    P = 32

    # Random bipolar patterns
    patterns = rng.choice(
        [-1.0, 1.0],
        size=(P, N)
    )

    # Hebbian matrix
    W = np.zeros((N, N), dtype=float)

    for mu in range(P):
        x = patterns[mu]
        W += np.outer(x, x)

    W /= float(N)

    np.fill_diagonal(W, 0.0)

    # Select target
    target = patterns[0]

    # Corrupt target
    noise = rng.normal(
        0.0,
        0.6,
        size=N
    )

    x0 = target + noise

    # Recall
    recalled = hz_recall(
        W,
        x0,
        steps=320,
        alpha=0.25,
        eps=0.10,
        lam=0.02
    )

    # Cosine similarity
    def cos_sim(a, b):

        na = np.linalg.norm(a) + 1e-9
        nb = np.linalg.norm(b) + 1e-9

        return float(
            np.dot(a, b) / (na * nb)
        )

    print(
        "Cosine similarity:",
        round(cos_sim(recalled, target), 4)
    )
