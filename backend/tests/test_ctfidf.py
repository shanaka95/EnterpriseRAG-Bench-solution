"""
Tests for the c-TF-IDF keyword extraction module.
"""
import pytest
import numpy as np

from app.ml.ctfidf import compute_ctfidf


class TestComputeCtfidf:
    """Tests for the compute_ctfidf function."""

    def test_basic_keyword_extraction(self, sample_cluster_texts):
        """Test that c-TF-IDF extracts meaningful keywords from distinct clusters."""
        result = compute_ctfidf(sample_cluster_texts, top_k=5)

        assert isinstance(result, dict)
        assert len(result) == 3
        # Each cluster should get some keywords
        for cid in sample_cluster_texts:
            assert cid in result
            assert isinstance(result[cid], list)
            assert len(result[cid]) <= 5

    def test_ml_cluster_gets_ml_keywords(self):
        """Test that an ML-focused cluster gets ML-related keywords."""
        cluster_texts = {
            "ml": "machine learning deep learning neural network training prediction model accuracy gradient",
            "db": "database query SQL index schema transaction performance optimization",
        }
        result = compute_ctfidf(cluster_texts, top_k=3)

        # The ML cluster should have "machine", "learning" or similar terms
        ml_keywords = result["ml"]
        assert len(ml_keywords) > 0
        # At least one keyword should relate to ML
        ml_related = any(
            kw in " ".join(ml_keywords)
            for kw in ["machine", "learning", "neural", "deep", "gradient", "training"]
        )
        assert ml_related, f"Expected ML-related keywords, got: {ml_keywords}"

    def test_db_cluster_gets_db_keywords(self):
        """Test that a database-focused cluster gets DB-related keywords."""
        cluster_texts = {
            "ml": "machine learning deep learning neural network training prediction model",
            "db": "database query SQL index schema transaction performance optimization",
        }
        result = compute_ctfidf(cluster_texts, top_k=3)

        db_keywords = result["db"]
        assert len(db_keywords) > 0
        db_related = any(
            kw in " ".join(db_keywords)
            for kw in ["database", "query", "sql", "index", "schema", "transaction"]
        )
        assert db_related, f"Expected DB-related keywords, got: {db_keywords}"

    def test_empty_input(self):
        """Test that empty input returns empty dict."""
        result = compute_ctfidf({})
        assert result == {}

    def test_single_cluster(self):
        """Test c-TF-IDF with only one cluster (IDF will be uniform)."""
        cluster_texts = {
            "only": "neural network deep learning machine learning training data",
        }
        result = compute_ctfidf(cluster_texts, top_k=3)

        assert "only" in result
        assert isinstance(result["only"], list)
        # With only one cluster, IDF is log(1 + 1/(1+1)) for all terms,
        # so top keywords are just the highest TF terms
        assert len(result["only"]) > 0

    def test_top_k_limits_keywords(self):
        """Test that top_k properly limits the number of keywords."""
        cluster_texts = {
            "c1": "alpha beta gamma delta epsilon zeta eta theta iota kappa",
            "c2": "one two three four five six seven eight nine ten eleven",
        }
        for k in [1, 2, 5, 10]:
            result = compute_ctfidf(cluster_texts, top_k=k)
            for cid in result:
                assert len(result[cid]) <= k

    def test_duplicate_text_across_clusters(self):
        """Test when clusters have overlapping text content."""
        shared_text = "common shared words appear in both"
        cluster_texts = {
            "c1": f"{shared_text} unique alpha beta",
            "c2": f"{shared_text} unique gamma delta",
        }
        result = compute_ctfidf(cluster_texts, top_k=5)

        # Both clusters should have keywords
        assert len(result["c1"]) > 0
        assert len(result["c2"]) > 0
        # The unique words should rank higher than shared words
        # because shared words have lower IDF

    def test_bigram_extraction(self):
        """Test that c-TF-IDF extracts bigrams when meaningful."""
        cluster_texts = {
            "ml": "machine learning is great machine learning models are powerful deep learning too",
            "other": "completely different topic with unrelated words and phrases",
        }
        result = compute_ctfidf(cluster_texts, top_k=10)

        # With ngram_range=(1,2), "machine learning" could appear as a bigram
        ml_keywords = result["ml"]
        # At minimum, we should get some keywords
        assert len(ml_keywords) > 0

    def test_stop_words_removed(self):
        """Test that English stop words are filtered out."""
        cluster_texts = {
            "c1": "the quick brown fox jumps over the lazy dog",
            "c2": "data processing pipeline handles large datasets efficiently",
        }
        result = compute_ctfidf(cluster_texts, top_k=10)

        # Stop words like "the", "over", "is" should not appear
        c1_keywords = result["c1"]
        stop_words = {"the", "over", "is", "and", "or", "in", "on", "at", "to", "for"}
        for kw in c1_keywords:
            assert kw not in stop_words, f"Stop word '{kw}' should not be a keyword"

    def test_numeric_cluster_ids(self):
        """Test that numeric cluster IDs work correctly."""
        cluster_texts = {
            "0": "machine learning neural network deep learning model",
            "1": "database query SQL index performance optimization",
            "2": "web development frontend backend API server",
        }
        result = compute_ctfidf(cluster_texts, top_k=5)

        assert len(result) == 3
        assert "0" in result
        assert "1" in result
        assert "2" in result

    def test_returns_empty_for_unparseable_text(self):
        """Test behavior when cluster text has no valid terms after vectorization."""
        # All numbers/symbols - CountVectorizer may produce empty vocabulary
        cluster_texts = {
            "c1": "123 456 789",
            "c2": "!!! ??? ###",
        }
        result = compute_ctfidf(cluster_texts, top_k=5)

        # Should return empty lists, not crash
        assert isinstance(result, dict)
        for cid in result:
            assert isinstance(result[cid], list)

    def test_cluster_with_single_word(self):
        """Test clusters with very short text."""
        cluster_texts = {
            "c1": "machine",
            "c2": "database",
        }
        result = compute_ctfidf(cluster_texts, top_k=3)

        assert "c1" in result
        assert "c2" in result

    def test_large_top_k(self):
        """Test with top_k larger than available terms."""
        cluster_texts = {
            "c1": "alpha beta",
            "c2": "gamma delta",
        }
        result = compute_ctfidf(cluster_texts, top_k=100)

        # Should return at most as many keywords as there are terms
        for cid in result:
            assert isinstance(result[cid], list)
