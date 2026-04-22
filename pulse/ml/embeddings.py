"""Embedding generation with a three-tier strategy:

1. OpenAI `text-embedding-3-small` when OPENAI_API_KEY is configured.
2. Sentence-Transformers `all-MiniLM-L6-v2` — real semantic embeddings,
   no API calls. ~90 MB model, downloaded to disk on first use. This is
   the default when no OpenAI key is present.
3. Deterministic TF-IDF hashing — last-resort fallback if
   sentence-transformers is not installed or fails to load. Weak for
   paraphrased text, sufficient for near-duplicates.
"""

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
_MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_MINILM_DIMENSIONS = 384

_minilm_model = None  # lazy-loaded
_minilm_load_failed = False


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


# ── Sentence-Transformers (MiniLM) ───────────────────────────────────────────


def _get_minilm():
    """Lazy-load the MiniLM model. Returns None if unavailable."""
    global _minilm_model, _minilm_load_failed
    if _minilm_model is not None:
        return _minilm_model
    if _minilm_load_failed:
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning(
            "sentence-transformers not installed; falling back to hashing embeddings. "
            "Install with: pip install sentence-transformers"
        )
        _minilm_load_failed = True
        return None
    try:
        _minilm_model = SentenceTransformer(_MINILM_MODEL_NAME)
        logger.info("Loaded local embedding model: %s", _MINILM_MODEL_NAME)
    except Exception as exc:
        logger.warning("Failed to load %s: %s", _MINILM_MODEL_NAME, exc)
        _minilm_load_failed = True
        return None
    return _minilm_model


def embed_texts_minilm(texts: list[str]) -> NDArray[np.float64]:
    """Generate embeddings via sentence-transformers MiniLM."""
    model = _get_minilm()
    if model is None:
        raise RuntimeError("MiniLM model not available")
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float64)


# ── TF-IDF hashing fallback (last resort, low quality for paraphrases) ───────


_GLOBAL_IDF: dict[str, float] = {}
_IDF_DEFAULT = 5.0


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1]


def _hash_token(token: str, dim: int) -> int:
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % dim


def embed_texts_hashing(texts: list[str], dim: int = 256) -> NDArray[np.float64]:
    """Deterministic TF-IDF hashing embeddings. No external dependency, low quality."""
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
            sign = 1 if int(hashlib.sha1(tok.encode()).hexdigest(), 16) % 2 == 0 else -1
            matrix[i, idx] += sign * weight

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix /= norms
    return matrix


# Backwards-compatible alias — some callers still import the old name.
def embed_texts_local(texts: list[str], dim: int | None = None) -> NDArray[np.float64]:
    """Return a local (no-API) embedding, preferring MiniLM over hashing."""
    if dim is None:
        dim = settings.embedding_dimensions
    try:
        return embed_texts_minilm(texts)
    except Exception as exc:
        logger.warning("MiniLM embed failed, falling back to hashing: %s", exc)
        return embed_texts_hashing(texts, dim=dim)


# ── Unified interface ────────────────────────────────────────────────────────


def embed_texts(texts: list[str]) -> NDArray[np.float64]:
    """Generate embeddings.

    Priority:
      1. OpenAI, if OPENAI_API_KEY is configured.
      2. Local MiniLM (sentence-transformers), always available when the
         package is installed — gives real semantic similarity for
         paraphrased feedback.
      3. Hashing TF-IDF, only if MiniLM fails to load.
    """
    if not texts:
        return np.zeros((0, settings.embedding_dimensions), dtype=np.float64)

    if settings.openai_api_key:
        try:
            return embed_texts_openai(texts)
        except Exception as exc:
            logger.warning("OpenAI embedding failed, falling back to local: %s", exc)

    try:
        return embed_texts_minilm(texts)
    except Exception as exc:
        logger.warning("MiniLM embedding failed, falling back to hashing: %s", exc)
        return embed_texts_hashing(texts, dim=settings.embedding_dimensions)


def get_embedding_info() -> dict[str, str | int]:
    """Report which embedding backend is active. Useful for /system/status."""
    if settings.openai_api_key:
        return {
            "provider": "openai",
            "model": settings.embedding_model,
            "dimensions": settings.embedding_dimensions,
        }
    # Don't actually load MiniLM just to report status; check for package presence.
    try:
        import sentence_transformers  # noqa: F401
        return {
            "provider": "sentence-transformers",
            "model": _MINILM_MODEL_NAME,
            "dimensions": _MINILM_DIMENSIONS,
        }
    except ImportError:
        return {
            "provider": "hashing",
            "model": "tf-idf-hash",
            "dimensions": settings.embedding_dimensions,
        }
