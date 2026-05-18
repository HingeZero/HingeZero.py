import numpy as np
from dataclasses import dataclass

# ============================================================
# HINGEZERO — SCALABLE PRODUCTION-STYLE SINGLE CELL
# No dense N x N matrix. Uses memory bank + Top-K + HZ refinement.
# ============================================================

def l2_normalize(x, axis=-1, eps=1e-9):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)

def hinge_phi(h, alpha=0.25):
    return np.tanh(h) + alpha * np.tanh(2.0 * h)

def cosine_similarity(a, b, eps=1e-9):
    return float(np.dot(a, b) / ((np.linalg.norm(a) + eps) * (np.linalg.norm(b) + eps)))

@dataclass
class RecallResult:
    index: int
    similarity: float
    margin: float
    recalled: np.ndarray
    topk_indices: np.ndarray
    topk_scores: np.ndarray

class HingeZeroMemory:
    def __init__(self, metric="cosine"):
        if metric not in ("cosine", "dot"):
            raise ValueError("metric must be 'cosine' or 'dot'")
        self.metric = metric
        self.memory = None
        self.memory_norm = None

    def fit(self, patterns):
        patterns = np.asarray(patterns, dtype=np.float32)
        if patterns.ndim != 2:
            raise ValueError("patterns must be shape (P, N)")
        self.memory = patterns.copy()
        self.memory_norm = l2_normalize(patterns, axis=1)
        return self

    def _scores(self, query):
        q = np.asarray(query, dtype=np.float32)
        if self.metric == "cosine":
            q = l2_normalize(q.reshape(1, -1), axis=1)[0]
            return self.memory_norm @ q
        return self.memory @ q

    def recall(
        self,
        query,
        top_k=32,
        steps=12,
        alpha=0.25,
        eps=0.10,
        lam=0.02,
        beta=8.0,
        normalize=True,
    ):
        if self.memory is None:
            raise RuntimeError("Call fit(patterns) first.")

        query = np.asarray(query, dtype=np.float32)
        scores = self._scores(query)

        k = min(int(top_k), len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]

        candidates = self.memory[idx].astype(np.float32)
        cand_norm = l2_normalize(candidates, axis=1)

        x = query.copy()
        if normalize:
            x = l2_normalize(x.reshape(1, -1), axis=1)[0]

        for _ in range(int(steps)):
            q = l2_normalize(x.reshape(1, -1), axis=1)[0]
            sims = cand_norm @ q

            z = beta * (sims - np.max(sims))
            weights = np.exp(z)
            weights /= np.sum(weights) + 1e-9

            h = weights @ candidates
            x = (1.0 - lam) * x + eps * hinge_phi(h, alpha=alpha)

            if normalize:
                x = l2_normalize(x.reshape(1, -1), axis=1)[0]

        final_scores = self._scores(x)
        order = np.argsort(-final_scores)

        best = int(order[0])
        second = int(order[1]) if len(order) > 1 else best
        margin = float(final_scores[best] - final_scores[second])

        return RecallResult(
            index=best,
            similarity=float(final_scores[best]),
            margin=margin,
            recalled=x,
            topk_indices=idx,
            topk_scores=scores[idx],
        )

# ============================================================
# QUALITY CONTROL TEST
# ============================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    N = 512
    P = 100_000
    top_k = 64
    trials = 100
    noise_sigma = 0.50

    print("=== HingeZero scalable production-style test ===")
    print(f"N={N} | P={P:,} | top_k={top_k} | trials={trials} | noise={noise_sigma}")

    memory = rng.choice([-1.0, 1.0], size=(P, N)).astype(np.float32)

    hz = HingeZeroMemory(metric="cosine")
    hz.fit(memory)

    correct = 0
    margins = []
    sims = []

    for t in range(trials):
        target_id = int(rng.integers(0, P))
        target = memory[target_id]

        query = target + rng.normal(0.0, noise_sigma, size=N).astype(np.float32)

        result = hz.recall(
            query,
            top_k=top_k,
            steps=12,
            alpha=0.25,
            eps=0.10,
            lam=0.02,
            beta=8.0,
            normalize=True,
        )

        hit = result.index == target_id
        correct += int(hit)
        margins.append(result.margin)
        sims.append(cosine_similarity(result.recalled, target))

        if t < 10:
            print(
                f"[{t+1:03d}] true={target_id} pred={result.index} "
                f"hit={hit} sim={sims[-1]:.4f} margin={result.margin:.6f}"
            )

    print("\n=== RESULTS ===")
    print("top1_accuracy :", round(correct / trials, 4))
    print("mean_similarity:", round(float(np.mean(sims)), 4))
    print("mean_margin    :", round(float(np.mean(margins)), 6))
    print("memory_MB      :", round(memory.nbytes / 1024 / 1024, 2))
