"""Smoke tests for the ML pipeline."""

import numpy as np

from pulse.ml.sentiment import analyze_sentiment, analyze_urgency, classify_category, enrich_feedback
from pulse.ml.embeddings import embed_texts_hashing, embed_texts_local
from pulse.ml.clustering import cluster_embeddings, extract_theme, extract_keywords, choose_cluster_count
from pulse.ml.anomaly import detect_volume_spike, detect_sentiment_shift


def test_sentiment_positive():
    assert analyze_sentiment("I love this, it's amazing!") > 0.3


def test_sentiment_negative():
    assert analyze_sentiment("This is terrible and broken") < -0.3


def test_sentiment_negation():
    assert analyze_sentiment("This is not great") < 0


def test_urgency():
    assert analyze_urgency("This is a critical blocker") >= 0.8
    assert analyze_urgency("nice to have someday") == 0.0


def test_classify_category():
    cat, _ = classify_category("The dashboard is slow and laggy")
    assert cat == "performance"

    cat, _ = classify_category("The app crashes on startup")
    assert cat == "bug"

    cat, _ = classify_category("Please add Salesforce integration")
    assert cat in ("feature-request", "integration")


def test_enrich_feedback_has_all_fields():
    result = enrich_feedback("The dashboard crashes and I'm blocked")
    assert "sentiment" in result
    assert "urgency" in result
    assert "category" in result
    assert result["urgency"] > 0.5


def test_embed_texts_hashing_shape():
    texts = ["the dashboard is slow", "search is broken", "love the new feature"]
    embeds = embed_texts_hashing(texts, dim=64)
    assert embeds.shape == (3, 64)
    # L2 normalised
    norms = np.linalg.norm(embeds, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_embed_texts_local_produces_normalised_embeddings():
    # Doesn't assert a specific dim — MiniLM is 384d, hashing is configurable,
    # the public contract is only "semantic-ish, row-normalised".
    texts = ["the dashboard is slow", "search is broken", "love the new feature"]
    embeds = embed_texts_local(texts)
    assert embeds.shape[0] == 3
    norms = np.linalg.norm(embeds, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_cluster_embeddings():
    texts = [
        "dashboard is slow",
        "loading time is terrible",
        "the app crashes constantly",
        "crash bug on startup",
    ]
    embeds = embed_texts_hashing(texts, dim=64)
    labels, sims, centroids = cluster_embeddings(embeds, n_clusters=2, similarity_threshold=0.0)
    assert len(labels) == 4
    assert centroids.shape[0] == 2


def test_extract_theme():
    texts = [
        "dashboard is slow to load",
        "dashboard loading performance is terrible",
        "slow latency on dashboard reports",
    ]
    theme = extract_theme(texts)
    assert "dashboard" in theme.lower() or "performance" in theme.lower()


def test_extract_keywords():
    texts = ["dashboard slow", "dashboard loading", "slow performance dashboard"]
    kws = extract_keywords(texts, top_n=3)
    assert "dashboard" in kws


def test_choose_cluster_count():
    assert choose_cluster_count(10) >= 3
    assert choose_cluster_count(1000) <= 20
    assert choose_cluster_count(2) == 2


def test_detect_volume_spike():
    # 7 days of baseline, then a spike
    counts = [10, 12, 11, 9, 13, 10, 11, 50]
    anomaly = detect_volume_spike(counts)
    assert anomaly is not None
    assert anomaly.type == "volume_spike"


def test_detect_sentiment_shift():
    # Stable positive, then drop
    sentiments = [0.3, 0.28, 0.32, 0.29, 0.31, 0.3, 0.29, -0.4]
    anomaly = detect_sentiment_shift(sentiments)
    assert anomaly is not None
    assert anomaly.severity in ("critical", "warning")
