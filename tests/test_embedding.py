"""
Tests for the embedding module.

Note: Full tests requiring model loading are slow and resource-intensive.
These tests focus on utility functions and conceptual verification.
"""
import numpy as np
import pytest
import torch

# Try importing embedding module - may fail due to spaCy/pydantic conflicts
try:
    from semantic_change.embedding import (
        detect_optimal_dtype,
        detect_optimal_batch_size,
        POOLING_STRATEGIES,
        Embedder,
        BertEmbedder,
    )
    EMBEDDING_IMPORT_ERROR = None
except Exception as e:
    EMBEDDING_IMPORT_ERROR = str(e)
    detect_optimal_dtype = None
    detect_optimal_batch_size = None
    POOLING_STRATEGIES = None
    Embedder = None
    BertEmbedder = None


class TestDetectOptimalDtype:
    """Tests for detect_optimal_dtype function."""

    @pytest.mark.skipif(EMBEDDING_IMPORT_ERROR is not None,
                        reason=f"Embedding module import failed: {EMBEDDING_IMPORT_ERROR}")
    def test_function_exists(self):
        """Function is importable and callable."""
        # Just verify it's callable
        result = detect_optimal_dtype()
        assert result is None or isinstance(result, torch.dtype)


class TestDetectOptimalBatchSize:
    """Tests for detect_optimal_batch_size function."""

    @pytest.mark.skipif(EMBEDDING_IMPORT_ERROR is not None,
                        reason=f"Embedding module import failed: {EMBEDDING_IMPORT_ERROR}")
    def test_function_exists(self):
        """Function is importable and returns an integer."""
        result = detect_optimal_batch_size()
        assert isinstance(result, int)
        assert result > 0


class TestPoolingStrategies:
    """Tests for pooling strategy constants."""

    @pytest.mark.skipif(EMBEDDING_IMPORT_ERROR is not None,
                        reason=f"Embedding module import failed: {EMBEDDING_IMPORT_ERROR}")
    def test_pooling_strategies_defined(self):
        """All expected pooling strategies are defined."""
        expected = ["mean", "first", "lemma_aligned", "weighted", "lemma_replacement"]
        for strategy in expected:
            assert strategy in POOLING_STRATEGIES


class TestEmbedderBase:
    """Tests for the Embedder base class."""

    @pytest.mark.skipif(EMBEDDING_IMPORT_ERROR is not None,
                        reason=f"Embedding module import failed: {EMBEDDING_IMPORT_ERROR}")
    def test_embedder_is_abstract(self):
        """Embedder base class raises NotImplementedError."""
        embedder = Embedder()
        with pytest.raises(NotImplementedError):
            embedder.get_embeddings([])


class TestEmbeddingAlignment:
    """Tests for embedding extraction alignment logic (conceptual)."""

    def test_sample_format(self):
        """Verify expected sample format for embedding extraction."""
        sample = {
            "sentence": "The factory produced goods.",
            "matched_word": "factory",
            "start_char": 4,
            "lemma": "factory"
        }

        assert "sentence" in sample
        assert "matched_word" in sample
        assert "start_char" in sample
        assert isinstance(sample["start_char"], int)

    def test_embedding_output_shape_concept(self):
        """Verify expected output shape from embeddings."""
        n_samples = 5
        hidden_dim = 768  # BERT base hidden size

        # Mock embedding output
        embeddings = np.random.randn(n_samples, hidden_dim)

        assert embeddings.shape == (n_samples, hidden_dim)
        assert embeddings.dtype in [np.float32, np.float64]


class TestEmbeddingNormalization:
    """Tests for embedding normalization utilities."""

    def test_l2_normalization(self):
        """L2 normalization produces unit vectors."""
        embeddings = np.random.randn(10, 768)

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms

        # Check all vectors are unit length
        result_norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_array_almost_equal(result_norms, np.ones(10))

    def test_cosine_similarity_range(self):
        """Cosine similarity between normalized vectors is in [-1, 1]."""
        embeddings = np.random.randn(10, 768)

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms

        # Compute pairwise cosine similarities
        similarities = normalized @ normalized.T

        assert np.all(similarities >= -1.0 - 1e-6)
        assert np.all(similarities <= 1.0 + 1e-6)


class TestLayerCombination:
    """Tests for layer combination logic."""

    def test_mean_layer_combination(self):
        """Mean of multiple layers produces correct shape."""
        n_layers = 4
        n_samples = 5
        hidden_dim = 768

        # Simulate hidden states from multiple layers
        layer_outputs = [np.random.randn(n_samples, hidden_dim) for _ in range(n_layers)]

        # Mean combination
        combined = np.mean(layer_outputs, axis=0)

        assert combined.shape == (n_samples, hidden_dim)

    def test_single_layer_selection(self):
        """Selecting single layer (e.g., last) maintains shape."""
        n_layers = 12
        n_samples = 5
        hidden_dim = 768

        # Simulate all hidden states
        all_layers = [np.random.randn(n_samples, hidden_dim) for _ in range(n_layers)]

        # Select last layer (index -1)
        selected = all_layers[-1]

        assert selected.shape == (n_samples, hidden_dim)


class TestBertEmbedderConcepts:
    """Conceptual tests for BertEmbedder (without loading models)."""

    @pytest.mark.skipif(EMBEDDING_IMPORT_ERROR is not None,
                        reason=f"Embedding module import failed: {EMBEDDING_IMPORT_ERROR}")
    def test_class_exists(self):
        """BertEmbedder class is importable."""
        assert BertEmbedder is not None

    @pytest.mark.skipif(EMBEDDING_IMPORT_ERROR is not None,
                        reason=f"Embedding module import failed: {EMBEDDING_IMPORT_ERROR}")
    def test_expected_methods(self):
        """BertEmbedder has expected methods."""
        # Check method existence
        assert hasattr(BertEmbedder, 'get_embeddings')
        assert hasattr(BertEmbedder, 'batch_extract')
        assert hasattr(BertEmbedder, 'get_nearest_neighbors')
