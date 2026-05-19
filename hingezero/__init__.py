"""HingeZero: Scalable associative memory with dual-tanh hinge operator.

A compact associative memory system using a smooth, deterministic dual-tanh hinge
operator for robust recall under noisy or corrupted inputs.
"""

from .memory import HingeZeroMemory, RecallResult
from .operators import hinge_phi, l2_normalize, cosine_similarity

__version__ = "0.1.0"
__author__ = "David Duffy"
__license__ = "GPL-3.0"

__all__ = [
    "HingeZeroMemory",
    "RecallResult",
    "hinge_phi",
    "l2_normalize",
    "cosine_similarity",
]
