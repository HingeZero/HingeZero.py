"""Tests for HingeZero memory system."""

import numpy as np
import pytest
from hingezero.memory import HingeZeroMemory, RecallResult


class TestRecallResult:
    """Tests for RecallResult dataclass."""

    def test_recall_result_creation(self):
        """Test RecallResult can be instantiated."""
        idx_array = np.array([0, 1, 2])
        score_array = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        recalled = np.array([0.1, 0.2], dtype=np.float32)

        result = RecallResult(
            index=0,
            similarity=0.95,
            margin=0.1,
            recalled=recalled,
            topk_indices=idx_array,
            topk_scores=score_array,
        )

        assert result.index == 0
        assert result.similarity == 0.95
        assert result.margin == 0.1
        np.testing.assert_array_equal(result.topk_indices, idx_array)


class TestHingeZeroMemoryInit:
    """Tests for HingeZeroMemory initialization."""

    def test_init_default(self):
        """Test default initialization."""
        hz = HingeZeroMemory()
        assert hz.metric == "cosine"
        assert hz.memory is None
        assert hz.memory_norm is None

    def test_init_dot_metric(self):
        """Test initialization with dot metric."""
        hz = HingeZeroMemory(metric="dot")
        assert hz.metric == "dot"

    def test_init_invalid_metric(self):
        """Test initialization fails with invalid metric."""
        with pytest.raises(ValueError, match="metric must be"):
            HingeZeroMemory(metric="euclidean")


class TestHingeZeroMemoryFit:
    """Tests for HingeZeroMemory.fit method."""

    def test_fit_2d_array(self):
        """Test fit with valid 2D array."""
        patterns = np.random.randn(100, 32).astype(np.float32)
        hz = HingeZeroMemory()
        result = hz.fit(patterns)

        assert result is hz  # Test method chaining
        assert hz.memory is not None
        assert hz.memory.shape == (100, 32)
        assert hz.memory_norm is not None
        assert hz.memory_norm.shape == (100, 32)

    def test_fit_copies_memory(self):
        """Test fit makes a copy of input."""
        patterns = np.random.randn(10, 8).astype(np.float32)
        hz = HingeZeroMemory()
        hz.fit(patterns)

        patterns[0, 0] = 999.0
        assert hz.memory[0, 0] != 999.0

    def test_fit_1d_array_fails(self):
        """Test fit fails with 1D array."""
        patterns = np.random.randn(32)
        hz = HingeZeroMemory()
        with pytest.raises(ValueError, match="must be 2D"):
            hz.fit(patterns)

    def test_fit_3d_array_fails(self):
        """Test fit fails with 3D array."""
        patterns = np.random.randn(10, 8, 4)
        hz = HingeZeroMemory()
        with pytest.raises(ValueError, match="must be 2D"):
            hz.fit(patterns)

    def test_fit_list_input(self):
        """Test fit accepts list input."""
        patterns = [[1.0, 0.0], [0.0, 1.0]]
        hz = HingeZeroMemory()
        hz.fit(patterns)
        assert hz.memory.shape == (2, 2)


class TestHingeZeroMemoryRecall:
    """Tests for HingeZeroMemory.recall method."""

    @pytest.fixture
    def setup_memory(self):
        """Create test memory."""
        rng = np.random.default_rng(42)
        patterns = rng.choice([-1.0, 1.0], size=(1000, 64)).astype(np.float32)
        hz = HingeZeroMemory(metric="cosine")
        hz.fit(patterns)
        return hz, patterns

    def test_recall_without_fit_fails(self):
        """Test recall fails if fit not called."""
        hz = HingeZeroMemory()
        query = np.ones(32, dtype=np.float32)
        with pytest.raises(RuntimeError, match="fit"):
            hz.recall(query)

    def test_recall_clean_query(self, setup_memory):
        """Test recall with clean (noiseless) query."""
        hz, patterns = setup_memory
        query = patterns[0].copy()
        result = hz.recall(query, top_k=32, steps=12)

        assert isinstance(result, RecallResult)
        assert result.index >= 0
        assert result.similarity >= -1.0 and result.similarity <= 1.0
        assert result.margin >= 0.0
        assert result.recalled.shape == query.shape

    def test_recall_noisy_query(self, setup_memory):
        """Test recall with noisy query."""
        hz, patterns = setup_memory
        rng = np.random.default_rng(42)
        target_idx = 42
        target = patterns[target_idx]
        query = target + rng.normal(0.0, 0.3, size=target.shape).astype(np.float32)

        result = hz.recall(query, top_k=32, steps=12)
        assert result.index >= 0

    def test_recall_returns_correct_fields(self, setup_memory):
        """Test recall returns all required fields."""
        hz, patterns = setup_memory
        query = patterns[0]
        result = hz.recall(query, top_k=16)

        assert hasattr(result, "index")
        assert hasattr(result, "similarity")
        assert hasattr(result, "margin")
        assert hasattr(result, "recalled")
        assert hasattr(result, "topk_indices")
        assert hasattr(result, "topk_scores")

    def test_recall_top_k_respected(self, setup_memory):
        """Test top_k parameter is respected."""
        hz, patterns = setup_memory
        query = patterns[0]
        result = hz.recall(query, top_k=8)

        assert len(result.topk_indices) <= 8
        assert len(result.topk_scores) <= 8

    def test_recall_dot_metric(self):
        """Test recall with dot product metric."""
        rng = np.random.default_rng(42)
        patterns = rng.normal(0, 1, size=(100, 32)).astype(np.float32)
        hz = HingeZeroMemory(metric="dot")
        hz.fit(patterns)

        query = patterns[0]
        result = hz.recall(query, top_k=16)
        assert result.index >= 0

    def test_recall_parameter_variations(self, setup_memory):
        """Test recall with different parameters."""
        hz, patterns = setup_memory
        query = patterns[0]

        # Different step counts
        result1 = hz.recall(query, steps=5)
        result2 = hz.recall(query, steps=20)
        assert result1.index >= 0 and result2.index >= 0

        # Different alpha values
        result3 = hz.recall(query, alpha=0.1)
        result4 = hz.recall(query, alpha=0.5)
        assert result3.index >= 0 and result4.index >= 0

    def test_recall_normalized_vs_unnormalized(self, setup_memory):
        """Test recall with and without normalization."""
        hz, patterns = setup_memory
        query = patterns[0]

        result_norm = hz.recall(query, normalize=True)
        result_no_norm = hz.recall(query, normalize=False)

        assert result_norm.recalled.shape == result_no_norm.recalled.shape


class TestHingeZeroMemoryAccuracy:
    """Integration tests for recall accuracy."""

    def test_clean_recall_accuracy(self):
        """Test recall accuracy with clean (noiseless) patterns."""
        rng = np.random.default_rng(42)
        patterns = rng.choice([-1.0, 1.0], size=(500, 128)).astype(np.float32)
        hz = HingeZeroMemory(metric="cosine")
        hz.fit(patterns)

        correct = 0
        trials = 50
        for i in range(trials):
            target_idx = rng.integers(0, len(patterns))
            result = hz.recall(
                patterns[target_idx],
                top_k=32,
                steps=12,
            )
            if result.index == target_idx:
                correct += 1

        accuracy = correct / trials
        assert accuracy > 0.8, f"Expected >80% accuracy, got {accuracy:.1%}"

    def test_noisy_recall_accuracy(self):
        """Test recall accuracy with noisy patterns."""
        rng = np.random.default_rng(42)
        patterns = rng.choice([-1.0, 1.0], size=(500, 128)).astype(np.float32)
        hz = HingeZeroMemory(metric="cosine")
        hz.fit(patterns)

        correct = 0
        trials = 50
        noise_sigma = 0.5

        for i in range(trials):
            target_idx = rng.integers(0, len(patterns))
            target = patterns[target_idx]
            query = target + rng.normal(0.0, noise_sigma, size=target.shape).astype(
                np.float32
            )
            result = hz.recall(
                query,
                top_k=32,
                steps=12,
            )
            if result.index == target_idx:
                correct += 1

        accuracy = correct / trials
        assert accuracy > 0.6, f"Expected >60% accuracy under noise, got {accuracy:.1%}"
