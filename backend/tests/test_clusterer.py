"""
Tests for the clustering module: UMAP reduction and HDBSCAN soft-assignment clustering.
Uses small synthetic datasets and CPU-only paths (no GPU required).
"""
import pytest
import numpy as np

from app.ml.clusterer import reduce_dimensions, cluster_with_soft_assignment, _hard_to_soft


class TestReduceDimensions:
    """Tests for the UMAP dimensionality reduction function."""

    def test_basic_reduction(self, sample_embeddings):
        """Test that UMAP reduces embedding dimensions correctly."""
        embeddings, _ = sample_embeddings
        n_components = 5
        reduced = reduce_dimensions(
            embeddings,
            n_neighbors=5,
            min_dist=0.1,
            n_components=n_components,
            metric="cosine",
        )

        assert reduced.shape == (60, n_components)
        assert reduced.dtype == np.float32

    def test_output_shape_matches_input_count(self, sample_embeddings):
        """Test that output has the same number of samples as input."""
        embeddings, _ = sample_embeddings
        reduced = reduce_dimensions(
            embeddings,
            n_neighbors=5,
            n_components=10,
        )

        assert reduced.shape[0] == embeddings.shape[0]

    def test_too_few_samples_returns_truncated(self):
        """Test that too-few samples return truncated/padded embeddings."""
        # 3 samples with 50 dims, n_neighbors=15 (too many)
        small_embeddings = np.random.rand(3, 50).astype(np.float32)
        result = reduce_dimensions(
            small_embeddings,
            n_neighbors=15,
            n_components=10,
        )

        # Should return shape (3, 10) - truncated from 50 to 10
        assert result.shape == (3, 10)

    def test_too_few_samples_pads_if_needed(self):
        """Test padding when embedding dim < n_components and too few samples."""
        small_embeddings = np.random.rand(3, 5).astype(np.float32)
        result = reduce_dimensions(
            small_embeddings,
            n_neighbors=15,
            n_components=10,
        )

        # Should return shape (3, 10) - padded from 5 to 10
        assert result.shape == (3, 10)
        # First 5 columns should match original, last 5 should be zeros
        np.testing.assert_array_almost_equal(result[:, :5], small_embeddings)
        np.testing.assert_array_almost_equal(result[:, 5:], np.zeros((3, 5)))

    def test_different_n_components(self, sample_embeddings):
        """Test reduction to various target dimensions."""
        embeddings, _ = sample_embeddings
        for n_comp in [2, 5, 10]:
            reduced = reduce_dimensions(
                embeddings,
                n_neighbors=5,
                n_components=n_comp,
            )
            assert reduced.shape == (60, n_comp)

    def test_different_metrics(self, sample_embeddings):
        """Test UMAP with different distance metrics."""
        embeddings, _ = sample_embeddings
        # cosine and euclidean should both work on CPU
        for metric in ["cosine", "euclidean"]:
            reduced = reduce_dimensions(
                embeddings,
                n_neighbors=5,
                n_components=5,
                metric=metric,
            )
            assert reduced.shape == (60, 5)

    def test_reproducibility_with_same_random_state(self, sample_embeddings):
        """Test that UMAP produces consistent results with the same random state."""
        embeddings, _ = sample_embeddings

        reduced1 = reduce_dimensions(
            embeddings, n_neighbors=5, n_components=5,
        )
        reduced2 = reduce_dimensions(
            embeddings, n_neighbors=5, n_components=5,
        )

        # With random_state=42 in the function, results should be deterministic
        np.testing.assert_array_almost_equal(reduced1, reduced2)


class TestClusterWithSoftAssignment:
    """Tests for HDBSCAN clustering with soft assignment."""

    def test_basic_clustering(self, sample_embeddings):
        """Test that HDBSCAN finds clusters in well-separated data."""
        embeddings, _ = sample_embeddings
        # First reduce dimensions
        reduced = reduce_dimensions(
            embeddings,
            n_neighbors=5,
            n_components=5,
        )
        cluster_to_docs, soft_probs, labels = cluster_with_soft_assignment(
            reduced,
            min_cluster_size=5,
            min_samples=3,
        )

        # Should find at least 1 cluster
        assert len(cluster_to_docs) >= 1
        # Labels should have same length as input
        assert len(labels) == len(reduced)
        # Soft probs should be 2D
        assert soft_probs.ndim == 2
        assert soft_probs.shape[0] == len(reduced)

    def test_cluster_to_docs_mapping(self, sample_embeddings):
        """Test that cluster_to_docs maps cluster labels to doc indices."""
        embeddings, _ = sample_embeddings
        reduced = reduce_dimensions(embeddings, n_neighbors=5, n_components=5)
        cluster_to_docs, _, labels = cluster_with_soft_assignment(
            reduced,
            min_cluster_size=5,
            min_samples=3,
        )

        # Every doc index in cluster_to_docs should be valid
        n_docs = len(reduced)
        for cluster_label, doc_indices in cluster_to_docs.items():
            for idx in doc_indices:
                assert 0 <= idx < n_docs

    def test_soft_probabilities_range(self, sample_embeddings):
        """Test that soft assignment probabilities are in [0, 1] range."""
        embeddings, _ = sample_embeddings
        reduced = reduce_dimensions(embeddings, n_neighbors=5, n_components=5)
        _, soft_probs, _ = cluster_with_soft_assignment(
            reduced,
            min_cluster_size=5,
            min_samples=3,
        )

        # Probabilities should be between 0 and 1
        assert soft_probs.min() >= -0.01  # Allow small numerical error
        assert soft_probs.max() <= 1.01

    def test_too_few_documents_single_cluster(self):
        """Test that too-few documents results in a single cluster."""
        small_embeddings = np.random.rand(4, 10).astype(np.float32)
        cluster_to_docs, soft_probs, labels = cluster_with_soft_assignment(
            small_embeddings,
            min_cluster_size=5,
            min_samples=3,
        )

        # Should assign all to single cluster 0
        assert 0 in cluster_to_docs
        assert len(cluster_to_docs[0]) == 4
        assert soft_probs.shape == (4, 1)
        np.testing.assert_array_almost_equal(soft_probs, np.ones((4, 1), dtype=np.float32))
        np.testing.assert_array_equal(labels, np.zeros(4, dtype=int))

    def test_soft_threshold_filters_members(self, sample_embeddings):
        """Test that soft_threshold parameter affects cluster membership."""
        embeddings, _ = sample_embeddings
        reduced = reduce_dimensions(embeddings, n_neighbors=5, n_components=5)

        # Low threshold: more docs should be included as soft members
        result_low, _, _ = cluster_with_soft_assignment(
            reduced, min_cluster_size=5, min_samples=3, soft_threshold=0.01,
        )
        total_low = sum(len(v) for v in result_low.values())

        # High threshold: fewer docs included
        result_high, _, _ = cluster_with_soft_assignment(
            reduced, min_cluster_size=5, min_samples=3, soft_threshold=0.99,
        )
        total_high = sum(len(v) for v in result_high.values())

        # Low threshold should include at least as many docs as high threshold
        assert total_low >= total_high

    def test_min_cluster_size_parameter(self, sample_embeddings):
        """Test that min_cluster_size affects the number of clusters found."""
        embeddings, _ = sample_embeddings
        reduced = reduce_dimensions(embeddings, n_neighbors=5, n_components=5)

        # Small min_cluster_size: may find more/smaller clusters
        result_small, _, _ = cluster_with_soft_assignment(
            reduced, min_cluster_size=3, min_samples=2,
        )
        # Large min_cluster_size: may find fewer/larger clusters
        result_large, _, _ = cluster_with_soft_assignment(
            reduced, min_cluster_size=10, min_samples=5,
        )

        # Both should find at least 1 cluster
        assert len(result_small) >= 1
        assert len(result_large) >= 1

    def test_labels_contain_negative_one_for_noise(self, sample_embeddings):
        """Test that noise points get label -1."""
        embeddings, _ = sample_embeddings
        reduced = reduce_dimensions(embeddings, n_neighbors=5, n_components=5)
        _, _, labels = cluster_with_soft_assignment(
            reduced, min_cluster_size=5, min_samples=3,
        )

        # Labels should contain -1 for noise and non-negative for clusters
        assert labels.min() >= -1
        # At least some points should be assigned to clusters
        assert (labels >= 0).any()


class TestHardToSoft:
    """Tests for the _hard_to_soft helper function."""

    def test_basic_conversion(self):
        """Test converting hard labels to soft probability matrix."""
        labels = np.array([0, 1, 0, 2, 1])
        soft = _hard_to_soft(labels)

        assert soft.shape == (5, 3)
        # Each row should sum to 0 or 1
        for i in range(5):
            assert soft[i].sum() <= 1.01

    def test_assigned_points_have_probability_one(self):
        """Test that hard-assigned points get probability 1.0."""
        labels = np.array([0, 1, 2])
        soft = _hard_to_soft(labels)

        assert soft[0, 0] == 1.0
        assert soft[1, 1] == 1.0
        assert soft[2, 2] == 1.0

    def test_noise_points_have_zero_probability(self):
        """Test that noise points (-1) get zero probability for all clusters."""
        labels = np.array([0, -1, 1])
        soft = _hard_to_soft(labels)

        # Noise point (index 1) should have all zeros
        np.testing.assert_array_equal(soft[1], np.zeros(2))

    def test_single_cluster(self):
        """Test conversion with only one cluster."""
        labels = np.array([0, 0, 0, -1])
        soft = _hard_to_soft(labels)

        assert soft.shape == (4, 1)
        assert soft[0, 0] == 1.0
        assert soft[3, 0] == 0.0  # noise

    def test_all_noise(self):
        """Test conversion when all points are noise."""
        labels = np.array([-1, -1, -1])
        soft = _hard_to_soft(labels)

        assert soft.shape == (3, 0)  # No clusters
        np.testing.assert_array_equal(soft, np.zeros((3, 0)))

    def test_non_contiguous_labels(self):
        """Test conversion with non-contiguous cluster labels."""
        labels = np.array([0, 5, 10])
        soft = _hard_to_soft(labels)

        # Should map labels 0->0, 5->1, 10->2
        assert soft.shape == (3, 3)
        assert soft[0, 0] == 1.0
        assert soft[1, 1] == 1.0
        assert soft[2, 2] == 1.0
