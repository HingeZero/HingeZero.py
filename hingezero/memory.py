"""HingeZero scalable associative memory implementation."""

import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional
from .operators import l2_normalize, hinge_phi


@dataclass
class RecallResult:
    """Result of a memory recall operation.

    Attributes
    ----------
    index : int
        Index of the best-matching memory pattern.
    similarity : float
        Similarity score of the recalled pattern to final query state.
    margin : float
        Margin between best and second-best match (confidence metric).
    recalled : np.ndarray
        Final recalled pattern after HingeZero refinement.
    topk_indices : np.ndarray
        Indices of top-K candidate patterns used in refinement.
    topk_scores : np.ndarray
        Similarity scores of top-K candidates.
    """
    index: int
    similarity: float
    margin: float
    recalled: np.ndarray
    topk_indices: np.ndarray
    topk_scores: np.ndarray


class HingeZeroMemory:
    """Scalable associative memory with HingeZero refinement.

    Uses a memory bank + top-K retrieval + HingeZero dynamics to perform
    robust associative recall under noise and corruption.

    Attributes
    ----------
    metric : {'cosine', 'dot'}
        Similarity metric for candidate search.
    memory : np.ndarray or None
        Memory bank of shape (P, N) containing P patterns of dimension N.
    memory_norm : np.ndarray or None
        L2-normalized memory bank (used for cosine metric).

    Examples
    --------
    >>> import numpy as np
    >>> from hingezero import HingeZeroMemory
    >>> rng = np.random.default_rng(0)
    >>> memory = rng.choice([-1, 1], size=(1000, 256))
    >>> hz = HingeZeroMemory(metric="cosine")
    >>> hz.fit(memory)
    >>> query = memory[0] + rng.normal(0, 0.5, size=256)
    >>> result = hz.recall(query, top_k=64)
    >>> print(f"Retrieved index: {result.index}, similarity: {result.similarity:.4f}")
    """

    def __init__(self, metric: Literal["cosine", "dot"] = "cosine") -> None:
        """Initialize HingeZero memory.

        Parameters
        ----------
        metric : {'cosine', 'dot'}, optional
            Similarity metric for candidate search. Default is 'cosine'.

        Raises
        ------
        ValueError
            If metric is not 'cosine' or 'dot'.
        """
        if metric not in ("cosine", "dot"):
            raise ValueError("metric must be 'cosine' or 'dot'")
        self.metric = metric
        self.memory: Optional[np.ndarray] = None
        self.memory_norm: Optional[np.ndarray] = None

    def fit(self, patterns: np.ndarray) -> "HingeZeroMemory":
        """Store patterns in memory bank.

        Parameters
        ----------
        patterns : np.ndarray
            Memory patterns of shape (P, N) where P is number of patterns
            and N is dimension.

        Returns
        -------
        HingeZeroMemory
            Self for method chaining.

        Raises
        ------
        ValueError
            If patterns is not 2D array.
        """
        patterns = np.asarray(patterns, dtype=np.float32)
        if patterns.ndim != 2:
            raise ValueError("patterns must be 2D array of shape (P, N)")
        self.memory = patterns.copy()
        self.memory_norm = l2_normalize(patterns, axis=1)
        return self

    def _scores(self, query: np.ndarray) -> np.ndarray:
        """Compute similarity scores between query and memory.

        Parameters
        ----------
        query : np.ndarray
            Query vector of shape (N,).

        Returns
        -------
        np.ndarray
            Similarity scores of shape (P,).
        """
        if self.memory is None:
            raise RuntimeError("Call fit(patterns) first")

        q = np.asarray(query, dtype=np.float32)
        if self.metric == "cosine":
            q = l2_normalize(q.reshape(1, -1), axis=1)[0]
            return self.memory_norm @ q
        return self.memory @ q

    def recall(
        self,
        query: np.ndarray,
        top_k: int = 32,
        steps: int = 12,
        alpha: float = 0.25,
        eps: float = 0.10,
        lam: float = 0.02,
        beta: float = 8.0,
        normalize: bool = True,
    ) -> RecallResult:
        """Recall memory pattern closest to noisy query.

        Performs iterative HingeZero refinement on top-K candidates.

        Parameters
        ----------
        query : np.ndarray
            Noisy or partial query vector of shape (N,).
        top_k : int, optional
            Number of candidate patterns to refine. Default is 32.
        steps : int, optional
            Number of HingeZero refinement iterations. Default is 12.
        alpha : float, optional
            Dual-tanh scaling parameter. Controls sensitivity around zero.
            Lower = wider equilibrium region. Default is 0.25.
        eps : float, optional
            Step size for HingeZero update. Default is 0.10.
        lam : float, optional
            Momentum/decay parameter for state update. Default is 0.02.
        beta : float, optional
            Temperature for softmax weighting of candidates. Default is 8.0.
        normalize : bool, optional
            Whether to L2-normalize state after each step. Default is True.

        Returns
        -------
        RecallResult
            Recall result containing best index, similarity, margin, and metadata.

        Raises
        ------
        RuntimeError
            If fit() has not been called.
        """
        if self.memory is None:
            raise RuntimeError("Call fit(patterns) first")

        query = np.asarray(query, dtype=np.float32)

        # Candidate search
        scores = self._scores(query)
        k = min(int(top_k), len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]

        candidates = self.memory[idx].astype(np.float32)
        cand_norm = l2_normalize(candidates, axis=1)

        # Initialize state
        x = query.copy()
        if normalize:
            x = l2_normalize(x.reshape(1, -1), axis=1)[0]

        # HingeZero refinement loop
        for _ in range(int(steps)):
            q = l2_normalize(x.reshape(1, -1), axis=1)[0]
            sims = cand_norm @ q

            # Softmax weighting
            z = beta * (sims - np.max(sims))
            weights = np.exp(z)
            weights /= np.sum(weights) + 1e-9

            # HingeZero update
            h = weights @ candidates
            x = (1.0 - lam) * x + eps * hinge_phi(h, alpha=alpha)

            if normalize:
                x = l2_normalize(x.reshape(1, -1), axis=1)[0]

        # Compute final scores and margin
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
