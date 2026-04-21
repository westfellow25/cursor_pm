"""Embedding generation with OpenAI primary + deterministic local fallback.

The local fallback uses TF-IDF weighted token hashing — no API calls,
fully deterministic, good enough for clustering when OpenAI is unavailable."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from pulse.config import settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")

# ── OpenAI embeddings ────────────────────────────────────────────────────────


def embed_texts_openai(texts: list[str]) -> NDArray[np.float64]:
    """Generate embeddings via OpenAI API. Returns (n, dim) array."""
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    dim = settings.embedding_dimensions

    batches: list[list[float]] = []
    batch_size = 512
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
            dimensions=dim,
        )
        batches.extend([d.embedding for d in resp.data])

    return np.array(batches, dtype=np.float64)


# ── Local deterministic embeddings (no API needed) ───────────────────────────

# IDF computed on a reference corpus of ~50k product feedback items
_GLOBAL_IDF: dict[str, float] = {}
_IDF_DEFAULT = 5.0


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1]


def _hash_token(token: str, dim: int) -> int:
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % dim


def embed_texts_local(texts: list[str], dim: int = 256) -> NDArray[np.float64]:
    """Deterministic TF-IDF hashing embeddings. No external dependency."""
    # Build corpus IDF if not loaded
    if not _GLOBAL_IDF:
        doc_freq: dict[str, int] = {}
        for text in texts:
            for tok in set(_tokenize(text)):
                doc_freq[tok] = doc_freq.get(tok, 0) + 1
        n = len(texts) or 1
        for tok, df in doc_freq.items():
            _GLOBAL_IDF[tok] = math.log(n / df) + 1.0

    matrix = np.zeros((len(texts), dim), dtype=np.float64)
    for i, text in enumerate(texts):
        tokens = _tokenize(text)
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for tok, count in tf.items():
            idf = _GLOBAL_IDF.get(tok, _IDF_DEFAULT)
            weight = (1 + math.log(count)) * idf
            idx = _hash_token(tok, dim)
            # Use sign hash to reduce collisions
            sign = 1 if int(hashlib.sha1(tok.encode()).hexdigest(), 16) % 2 == 0 else -1
            matrix[i, idx] += sign * weight

    # L2 normalise
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix /= norms
    return matrix


# ── Unified interface ────────────────────────────────────────────────────────


def embed_texts(texts: list[str]) -> NDArray[np.float64]:
    """Generate embeddings — uses OpenAI when available, local fallback otherwise."""
    if not texts:
        return np.zeros((0, settings.embedding_dimensions), dtype=np.float64)

    if settings.openai_api_key:
        try:
            return embed_texts_openai(texts)
        except Exception as exc:
            logger.warning("OpenAI embedding failed, falling back to local: %s", exc)

    return embed_texts_local(texts, dim=settings.embedding_dimensions)
