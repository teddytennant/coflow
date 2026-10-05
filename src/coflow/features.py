"""Handcrafted prompt features and a deterministic hash embedding.

Eight scalar features, in the paper's table order:

1. word count: min(#words / 50, 1)
2. type-token ratio: |unique lowercased tokens| / #tokens
3. mean word length: min(mean(|w|) / 10, 1)
4. numeric fraction: #words containing a digit / #words
5. object density: #noun-like tokens / #words, capped at 1
6. attribute density: #adjective-like tokens / #words, capped at 1
7. punctuation density: min(#punctuation / (2 * #words), 1)
8. syllable complexity: min(mean syllables / 4, 1)

Nouns are approximated as non-stopword tokens that contain a letter.
Adjectives are approximated as tokens that end with a common adjective
suffix. A token can count as both. This does not use a POS tagger.

The semantic embedding is not all-MiniLM-L6-v2. It is a bag of character
3-grams hashed into a fixed number of buckets, projected by a Gaussian
matrix from numpy.random.default_rng(0) to 32 dimensions, then L2-normalized.
"""

from __future__ import annotations

import re
import string

import numpy as np

HANDCRAFTED_DIM = 8
EMBED_DIM = 32
NGRAM = 3
NGRAM_BUCKETS = 512
EMBED_SEED = 0
CONTEXT_DIM = HANDCRAFTED_DIM + EMBED_DIM

FEATURE_NAMES = (
    "word_count",
    "ttr",
    "mean_word_length",
    "numeric_fraction",
    "object_density",
    "attribute_density",
    "punctuation_density",
    "syllable_complexity",
)

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")

# Common English function words. Used only for the noun approximation.
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "into",
        "over",
        "under",
        "than",
        "then",
        "so",
        "not",
        "no",
        "nor",
        "too",
        "very",
        "can",
        "will",
        "just",
        "about",
        "up",
        "out",
        "off",
        "above",
        "below",
        "between",
        "through",
        "during",
        "before",
        "after",
        "again",
        "further",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "both",
        "any",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "do",
        "does",
        "did",
        "doing",
        "have",
        "has",
        "had",
        "having",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "our",
        "their",
    }
)

# Longer suffixes first so the boolean check is easy to read. Matching is
# by endswith, so order does not change the result.
ADJECTIVE_SUFFIXES = (
    "esque",
    "able",
    "ible",
    "ious",
    "eous",
    "ical",
    "less",
    "like",
    "ous",
    "ful",
    "ive",
    "ish",
    "ary",
    "ory",
    "ent",
    "ant",
    "ic",
    "al",
    "y",
)

_PROJECTION_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def tokenize(text: str) -> list[str]:
    """Alphanumeric word tokens. Internal apostrophes stay inside a token."""
    if not text:
        return []
    return WORD_RE.findall(text)


def punctuation_count(text: str) -> int:
    if not text:
        return 0
    return sum(1 for ch in text if ch in string.punctuation)


def is_adjective(word: str) -> bool:
    """True if the token ends with a common adjective suffix and has a stem.

    The suffix 'y' is only accepted for tokens of length at least 5, so
    short nouns such as 'sky' are not marked as adjectives.
    """
    w = word.lower()
    for suf in ADJECTIVE_SUFFIXES:
        if not w.endswith(suf):
            continue
        if len(w) <= len(suf):
            continue
        if suf == "y" and len(w) < 5:
            continue
        return True
    return False


def is_noun(word: str) -> bool:
    """Non-stopword token that contains a letter.

    Pure numbers are not treated as nouns. Adjective-like tokens are still
    nouns under this rule if they are not stopwords.
    """
    w = word.lower()
    if w in STOPWORDS:
        return False
    return any(ch.isalpha() for ch in w)


def count_syllables(word: str) -> int:
    """Vowel-group count with a silent-e adjustment. At least 1."""
    letters = "".join(ch for ch in word.lower() if ch.isalpha())
    if not letters:
        return 1
    vowels = set("aeiouy")
    groups = 0
    prev = False
    for ch in letters:
        cur = ch in vowels
        if cur and not prev:
            groups += 1
        prev = cur
    if letters.endswith("e") and not letters.endswith("le") and groups > 1:
        groups -= 1
    return max(groups, 1)


def handcrafted_features(text: str) -> np.ndarray:
    """Return the 8 handcrafted features as a float64 vector in [0, 1]."""
    words = tokenize(text)
    n = len(words)
    feats = np.zeros(HANDCRAFTED_DIM, dtype=np.float64)
    if n == 0:
        return feats
    lower = [w.lower() for w in words]
    feats[0] = min(n / 50.0, 1.0)
    feats[1] = len(set(lower)) / n
    feats[2] = min((sum(len(w) for w in words) / n) / 10.0, 1.0)
    feats[3] = sum(1 for w in words if any(ch.isdigit() for ch in w)) / n
    feats[4] = min(sum(1 for w in words if is_noun(w)) / n, 1.0)
    feats[5] = min(sum(1 for w in words if is_adjective(w)) / n, 1.0)
    feats[6] = min(punctuation_count(text) / (2.0 * n), 1.0)
    feats[7] = min((sum(count_syllables(w) for w in words) / n) / 4.0, 1.0)
    return feats


def _fnv1a(text: str) -> int:
    h = 2166136261
    for ch in text:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def embedding_projection(n_buckets=NGRAM_BUCKETS, dim=EMBED_DIM, seed=EMBED_SEED):
    """Gaussian projection matrix from numpy Generator(seed), shape (buckets, dim)."""
    key = (int(n_buckets), int(dim), int(seed))
    cached = _PROJECTION_CACHE.get(key)
    if cached is None:
        rng = np.random.default_rng(key[2])
        cached = rng.normal(size=(key[0], key[1])).astype(np.float64)
        _PROJECTION_CACHE[key] = cached
    return cached


def hash_embedding(
    text: str,
    dim=EMBED_DIM,
    ngram=NGRAM,
    n_buckets=NGRAM_BUCKETS,
    seed=EMBED_SEED,
) -> np.ndarray:
    """L2-normalized projection of character n-gram counts.

    The prompt is lowercased. Grams shorter than n are produced by padding
    the string with spaces. An empty projection (all-zero counts) stays zero.
    """
    dim = int(dim)
    ngram = int(ngram)
    n_buckets = int(n_buckets)
    if not text:
        return np.zeros(dim, dtype=np.float64)
    raw = text.lower()
    if len(raw) < ngram:
        raw = raw + (" " * (ngram - len(raw)))
    counts = np.zeros(n_buckets, dtype=np.float64)
    for i in range(len(raw) - ngram + 1):
        counts[_fnv1a(raw[i : i + ngram]) % n_buckets] += 1.0
    projected = counts @ embedding_projection(n_buckets, dim, seed)
    norm = float(np.linalg.norm(projected))
    if norm == 0.0:
        return projected
    return projected / norm


def context_features(text: str) -> np.ndarray:
    """Concatenate the 8 handcrafted features and the 32-d hash embedding."""
    return np.concatenate([handcrafted_features(text), hash_embedding(text)])
