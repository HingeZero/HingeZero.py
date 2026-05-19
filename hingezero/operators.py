"""Core HingeZero operators and utility functions."""

import numpy as np
from typing import Union, Tuple


def l2_normalize(
    x: Union[np.ndarray, list],
    axis: int = -1,
    eps: float = 1e-9,
) -> np.ndarray:
    """L2 normalize array along specified axis.

    Parameters
    ----------
    x : array-like
        Input array to normalize.
    axis : int, optional
        Axis along which to normalize. Default is -1 (last axis).
    eps : float, optional
        Small epsilon to avoid division by zero. Default is 1e-9.

    Returns
    -------
    np.ndarray
        L2-normalized array in float32.
    """
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def hinge_phi(
    h: Union[np.ndarray, float],
    alpha: float = 0.25,
) -> Union[np.ndarray, float]:
    """HingeZero dual-tanh operator.

    The operator combines two tanh terms:
    - tanh(h): broad smooth stabilisation behaviour
    - α·tanh(2h): local shaping mechanism around zero

    Parameters
    ----------
    h : array-like or float
        Hidden state or field to apply operator to.
    alpha : float, optional
        Scaling factor for the second tanh term. Controls sensitivity around zero.
        Lower values = wider equilibrium region. Default is 0.25.

    Returns
    -------
    array-like or float
        Result of φ(h) = tanh(h) + α·tanh(2h)
    """
    return np.tanh(h) + alpha * np.tanh(2.0 * h)


def cosine_similarity(
    a: Union[np.ndarray, list],
    b: Union[np.ndarray, list],
    eps: float = 1e-9,
) -> float:
    """Compute cosine similarity between two vectors.

    Parameters
    ----------
    a : array-like
        First vector.
    b : array-like
        Second vector.
    eps : float, optional
        Small epsilon to avoid division by zero. Default is 1e-9.

    Returns
    -------
    float
        Cosine similarity score in range [-1, 1].
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / ((np.linalg.norm(a) + eps) * (np.linalg.norm(b) + eps)))
