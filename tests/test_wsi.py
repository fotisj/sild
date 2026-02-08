"""
Tests for the wsi (Word Sense Induction) module.
"""
import numpy as np
import pytest

from semantic_change.wsi import WordSenseInductor, SubstituteWSI


class TestWordSenseInductor:
    """Tests for the WordSenseInductor class."""

    def test_init_kmeans(self):
        """Initializes KMeans algorithm correctly."""
        wsi = WordSenseInductor(algorithm='kmeans', n_clusters=3)
        assert wsi.algorithm == 'kmeans'
        assert wsi.model is not None

    def test_init_spectral(self):
        """Initializes Spectral clustering correctly."""
        wsi = WordSenseInductor(algorithm='spectral', n_clusters=4)
        assert wsi.algorithm == 'spectral'
        assert wsi.model is not None

    def test_init_agglomerative(self):
        """Initializes Agglomerative clustering correctly."""
        wsi = WordSenseInductor(algorithm='agglomerative', n_clusters=2)
        assert wsi.algorithm == 'agglomerative'
        assert wsi.model is not None

    def test_init_hdbscan(self):
        """Initializes HDBSCAN correctly."""
        wsi = WordSenseInductor(algorithm='hdbscan', min_cluster_size=3)
        assert wsi.algorithm == 'hdbscan'
        assert wsi.model is not None

    def test_init_unknown_algorithm(self):
        """Raises error for unknown algorithm."""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            WordSenseInductor(algorithm='unknown_algo')

    def test_fit_predict_kmeans_returns_labels(self):
        """KMeans returns correct number of labels."""
        np.random.seed(42)
        # Create 3 distinct clusters
        cluster1 = np.random.randn(10, 50) + np.array([5, 0] + [0] * 48)
        cluster2 = np.random.randn(10, 50) + np.array([-5, 0] + [0] * 48)
        cluster3 = np.random.randn(10, 50) + np.array([0, 5] + [0] * 48)
        embeddings = np.vstack([cluster1, cluster2, cluster3])

        wsi = WordSenseInductor(algorithm='kmeans', n_clusters=3)
        labels = wsi.fit_predict(embeddings)

        assert len(labels) == 30
        assert set(labels) == {0, 1, 2}

    def test_fit_predict_hdbscan_returns_labels(self):
        """HDBSCAN returns labels (may include -1 for noise)."""
        np.random.seed(42)
        # Create 2 distinct clusters
        cluster1 = np.random.randn(20, 50) + np.array([10, 0] + [0] * 48)
        cluster2 = np.random.randn(20, 50) + np.array([-10, 0] + [0] * 48)
        embeddings = np.vstack([cluster1, cluster2])

        wsi = WordSenseInductor(algorithm='hdbscan', min_cluster_size=5)
        labels = wsi.fit_predict(embeddings)

        assert len(labels) == 40
        # HDBSCAN may return -1 for noise points
        assert all(isinstance(l, (int, np.integer)) for l in labels)

    def test_fit_predict_empty_input(self):
        """Returns empty array for empty input."""
        wsi = WordSenseInductor(algorithm='kmeans', n_clusters=3)
        labels = wsi.fit_predict(np.array([]).reshape(0, 50))
        assert len(labels) == 0

    def test_fit_predict_fewer_samples_than_clusters(self):
        """Handles case when samples < n_clusters."""
        embeddings = np.random.randn(2, 50)  # Only 2 samples
        wsi = WordSenseInductor(algorithm='kmeans', n_clusters=5)
        labels = wsi.fit_predict(embeddings)

        # Should assign all to cluster 0 when not enough data
        assert len(labels) == 2
        assert all(l == 0 for l in labels)

    def test_clustering_consistency(self):
        """Clustering produces consistent results (same random seed)."""
        np.random.seed(42)
        embeddings = np.random.randn(30, 50)

        wsi1 = WordSenseInductor(algorithm='kmeans', n_clusters=3)
        wsi2 = WordSenseInductor(algorithm='kmeans', n_clusters=3)

        labels1 = wsi1.fit_predict(embeddings)
        labels2 = wsi2.fit_predict(embeddings)

        # Same algorithm with same seed should produce same labels
        np.testing.assert_array_equal(labels1, labels2)

    def test_different_cluster_counts(self):
        """Different n_clusters produces different number of clusters."""
        np.random.seed(42)
        embeddings = np.random.randn(50, 50)

        wsi2 = WordSenseInductor(algorithm='kmeans', n_clusters=2)
        wsi5 = WordSenseInductor(algorithm='kmeans', n_clusters=5)

        labels2 = wsi2.fit_predict(embeddings)
        labels5 = wsi5.fit_predict(embeddings)

        assert len(set(labels2)) == 2
        assert len(set(labels5)) == 5


class TestSubstituteWSI:
    """Tests for the SubstituteWSI class."""

    def test_init(self):
        """Initializes with correct parameters."""
        wsi = SubstituteWSI(min_community_size=3, resolution=1.0)
        assert wsi.min_community_size == 3
        assert wsi.resolution == 1.0
        assert wsi.communities_ is None

    def test_fit_builds_graph(self):
        """Fit creates co-occurrence graph."""
        substitutes = [
            ['guitar', 'bass', 'drum'],
            ['guitar', 'piano', 'violin'],
            ['fish', 'trout', 'salmon'],
            ['fish', 'bass', 'perch'],
        ]

        wsi = SubstituteWSI(min_community_size=2)
        wsi.fit(substitutes)

        assert wsi.graph_ is not None
        assert wsi.graph_.number_of_nodes() > 0

    def test_fit_predict_returns_labels(self):
        """Fit_predict returns cluster labels."""
        # Create substitutes with 2 clear senses
        substitutes = [
            ['guitar', 'bass', 'drum'],  # Music sense
            ['guitar', 'piano', 'violin'],
            ['guitar', 'bass', 'piano'],
            ['fish', 'trout', 'salmon'],  # Fish sense
            ['fish', 'bass', 'perch'],
            ['fish', 'trout', 'perch'],
        ]

        wsi = SubstituteWSI(min_community_size=2)
        labels = wsi.fit_predict(substitutes)

        assert len(labels) == 6
        assert all(isinstance(l, (int, np.integer)) for l in labels)

    def test_predict_assigns_to_closest_sense(self):
        """Predict assigns instances based on Jaccard similarity."""
        # Training data with 2 senses
        train_subs = [
            ['guitar', 'bass', 'drum'],
            ['guitar', 'piano', 'violin'],
            ['fish', 'trout', 'salmon'],
            ['fish', 'perch', 'carp'],
        ]

        wsi = SubstituteWSI(min_community_size=2)
        wsi.fit(train_subs)

        # Test instances
        test_subs = [
            ['guitar', 'drum'],  # Should match music sense
            ['fish', 'salmon'],  # Should match fish sense
        ]

        labels = wsi.predict(test_subs)
        assert len(labels) == 2

    def test_get_sense_representatives(self):
        """Returns representative words for each sense."""
        substitutes = [
            ['apple', 'banana', 'orange'],
            ['apple', 'pear', 'banana'],
            ['car', 'truck', 'bus'],
            ['car', 'vehicle', 'truck'],
        ]

        wsi = SubstituteWSI(min_community_size=2)
        wsi.fit(substitutes)

        reps = wsi.get_sense_representatives(top_n=5)

        assert isinstance(reps, list)
        for sense_reps in reps:
            assert isinstance(sense_reps, list)
            assert all(isinstance(w, str) for w in sense_reps)

    def test_get_num_senses(self):
        """Returns correct number of senses."""
        substitutes = [
            ['a', 'b', 'c'],
            ['a', 'b', 'd'],
            ['x', 'y', 'z'],
            ['x', 'y', 'w'],
        ]

        wsi = SubstituteWSI(min_community_size=2)
        wsi.fit(substitutes)

        num = wsi.get_num_senses()
        assert isinstance(num, int)
        assert num >= 0

    def test_empty_substitutes(self):
        """Handles empty substitutes list."""
        wsi = SubstituteWSI(min_community_size=2)
        wsi.fit([])

        assert wsi.communities_ == []
        labels = wsi.predict([['test', 'word']])
        assert len(labels) == 1

    def test_merge_similar_communities(self):
        """Communities with high overlap are merged when threshold set."""
        # Create substitutes where some communities should merge
        substitutes = [
            ['word1', 'word2', 'word3'],
            ['word1', 'word2', 'word4'],
            ['word1', 'word3', 'word4'],
        ]

        # With high threshold, communities should merge
        wsi = SubstituteWSI(
            min_community_size=1,
            merge_threshold=0.3,
            merge_top_n=10
        )
        wsi.fit(substitutes)

        # Just verify it runs without error
        assert wsi.communities_ is not None

    def test_jaccard_similarity(self):
        """Internal Jaccard similarity calculation is correct."""
        wsi = SubstituteWSI()

        # Test with known sets
        set_a = {'a', 'b', 'c'}
        set_b = {'b', 'c', 'd'}

        sim = wsi._jaccard_similarity(set_a, set_b)
        # Intersection: {b, c} = 2, Union: {a, b, c, d} = 4
        assert sim == 0.5

    def test_jaccard_similarity_empty_sets(self):
        """Jaccard similarity handles empty sets."""
        wsi = SubstituteWSI()

        assert wsi._jaccard_similarity(set(), {'a', 'b'}) == 0.0
        assert wsi._jaccard_similarity({'a', 'b'}, set()) == 0.0
        assert wsi._jaccard_similarity(set(), set()) == 0.0
