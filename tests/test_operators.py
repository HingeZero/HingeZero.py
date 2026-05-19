"""Tests for HingeZero operators."""

import numpy as np
import pytest
from hingezero.operators import l2_normalize, hinge_phi, cosine_similarity


class TestL2Normalize:
    """Tests for l2_normalize function."""

    def test_normalize_single_vector(self):
        """Test L2 normalization of single vector."""
        x = np.array([3.0, 4.0])
        result = l2_normalize(x)
        expected = np.array([0.6, 0.8], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_normalize_matrix_axis0(self):
        """Test L2 normalization along axis 0."""
        x = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
        result = l2_normalize(x, axis=0)
        # First column: [3, 0] -> [1, 0]
        # Second column: [4, 5] -> [0.6, 0.8]
        expected = np.array([[1.0, 0.6], [0.0, 0.8]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_normalize_matrix_axis1(self):
        """Test L2 normalization along axis 1."""
        x = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
        result = l2_normalize(x, axis=1)
        expected = np.array([[0.6, 0.8], [0.0, 1.0]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_normalize_zero_vector(self):
        """Test normalization doesn't fail on zero vector."""
        x = np.array([0.0, 0.0])
        result = l2_normalize(x)
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()

    def test_normalize_list_input(self):
        """Test L2 normalization accepts list input."""
        x = [3.0, 4.0]
        result = l2_normalize(x)
        expected = np.array([0.6, 0.8], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_normalize_dtype_float32(self):
        """Test normalized output is float32."""
        x = np.array([1.0, 2.0], dtype=np.float64)
        result = l2_normalize(x)
        assert result.dtype == np.float32


class TestHingePhi:
    """Tests for hinge_phi operator."""

    def test_hinge_phi_zero(self):
        """Test hinge_phi at h=0."""
        result = hinge_phi(0.0)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_hinge_phi_positive(self):
        """Test hinge_phi with positive input."""
        h = 1.0
        alpha = 0.25
        expected = np.tanh(h) + alpha * np.tanh(2.0 * h)
        result = hinge_phi(h, alpha=alpha)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_hinge_phi_negative(self):
        """Test hinge_phi with negative input."""
        h = -1.0
        alpha = 0.25
        expected = np.tanh(h) + alpha * np.tanh(2.0 * h)
        result = hinge_phi(h, alpha=alpha)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_hinge_phi_array(self):
        """Test hinge_phi with array input."""
        h = np.array([-1.0, 0.0, 1.0])
        alpha = 0.25
        expected = np.tanh(h) + alpha * np.tanh(2.0 * h)
        result = hinge_phi(h, alpha=alpha)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_hinge_phi_alpha_effect(self):
        """Test that alpha parameter affects output."""
        h = 0.5
        result_alpha0 = hinge_phi(h, alpha=0.0)
        result_alpha1 = hinge_phi(h, alpha=1.0)
        assert result_alpha0 != result_alpha1
        assert result_alpha1 > result_alpha0  # Alpha=1 adds positive term


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        result = cosine_similarity(a, b)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        result = cosine_similarity(a, b)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity of opposite vectors."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        result = cosine_similarity(a, b)
        assert result == pytest.approx(-1.0, abs=1e-6)

    def test_cosine_similarity_scaled(self):
        """Test cosine similarity is scale-invariant."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 4.0, 6.0])
        result = cosine_similarity(a, b)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_cosine_similarity_list_input(self):
        """Test cosine_similarity accepts list inputs."""
        a = [1.0, 0.0]
        b = [1.0, 0.0]
        result = cosine_similarity(a, b)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_cosine_similarity_zero_vector(self):
        """Test cosine_similarity doesn't fail on zero vectors."""
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        result = cosine_similarity(a, b)
        assert not np.isnan(result)
        assert not np.isinf(result)
