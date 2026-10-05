"""Handcrafted features and the seed-0 hash embedding."""

import numpy as np

from coflow.features import (
    CONTEXT_DIM,
    EMBED_DIM,
    FEATURE_NAMES,
    HANDCRAFTED_DIM,
    context_features,
    embedding_projection,
    handcrafted_features,
    hash_embedding,
    is_adjective,
    is_noun,
    punctuation_count,
    tokenize,
)


def test_feature_formulas():
    text = "Cat cat, 3 dogs!"
    words = tokenize(text)
    assert words == ["Cat", "cat", "3", "dogs"]
    feats = handcrafted_features(text)
    assert feats.shape == (HANDCRAFTED_DIM,)
    assert len(FEATURE_NAMES) == 8
    n = 4
    assert feats[0] == min(n / 50.0, 1.0)
    assert feats[1] == 3 / 4  # cat, 3, dogs
    assert feats[2] == min(((3 + 3 + 1 + 4) / n) / 10.0, 1.0)
    assert feats[3] == 1 / 4
    assert feats[6] == min(punctuation_count(text) / (2 * n), 1.0)
    assert 0.0 <= feats.min() and feats.max() <= 1.0


def test_caps_and_empty():
    long = " ".join(["word"] * 60)
    feats = handcrafted_features(long)
    assert feats[0] == 1.0
    empty = handcrafted_features("")
    assert np.allclose(empty, 0.0)
    assert np.allclose(hash_embedding(""), 0.0)
    assert context_features("a cat").shape == (CONTEXT_DIM,)


def test_noun_and_adjective_approximation():
    assert is_noun("cat")
    assert not is_noun("the")
    assert not is_noun("3")
    assert is_adjective("beautiful")
    assert is_adjective("national")
    assert not is_adjective("sky")
    text = "the beautiful cat"
    feats = handcrafted_features(text)
    # the is a stopword; beautiful and cat are nouns; beautiful is an adjective.
    assert feats[4] == 2 / 3
    assert feats[5] == 1 / 3


def test_hash_embedding_is_deterministic_projection():
    text = "a red cube on a table"
    first = hash_embedding(text)
    second = hash_embedding(text)
    assert first.shape == (EMBED_DIM,)
    assert np.allclose(first, second)
    assert np.isclose(np.linalg.norm(first), 1.0)
    assert not np.allclose(first, hash_embedding("a blue sphere on a chair"))
    proj = embedding_projection()
    expected = np.random.default_rng(0).normal(size=proj.shape)
    assert np.allclose(proj, expected)
