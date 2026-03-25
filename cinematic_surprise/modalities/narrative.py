"""
modalities/narrative.py

Narrative feature extractor using sentence-transformers.

For each 1-second window, collects words from the transcript whose
start_time falls within [t, t+1), embeds the concatenated text as
a single sentence vector, and passes it to EMA+KL.

When no words are spoken (silence, music, no speech), the embedding
is not updated and the surprise is zero. This is the correct behaviour:
sustained silence is not narratively surprising.

Brain analog: Broca's area (left IFG), Wernicke's area (left STG/MTG),
              angular gyrus.

IN:  text string (words for this second, may be empty)
OUT: dict {"narrative": (384,)} float32  (or empty dict if unavailable)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    _ST_OK = True
except ImportError:
    _ST_OK = False
    log.warning(
        "sentence-transformers not installed. Narrative modality unavailable. "
        "Install with: pip install sentence-transformers"
    )

# Default model: all-MiniLM-L6-v2 → 384-d embeddings, fast, good quality
_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class NarrativeExtractor:
    """
    Sentence embedding extractor.

    Maintains the previous embedding internally. When a window has no
    speech, returns the previous embedding unchanged so that EMA+KL
    produces zero surprise (no change to belief).

    Args:
        model_name : sentence-transformers model name.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self.available = _ST_OK
        self._prev_embedding: Optional[np.ndarray] = None

        if not _ST_OK:
            return

        log.info(f"Loading sentence-transformers model '{model_name}'...")
        self._model = SentenceTransformer(model_name)
        log.info(f"Narrative extractor ready. Embedding dim: 384")

    def extract(self, text: str) -> Dict[str, np.ndarray]:
        if not self.available:
            return {}

        if text.strip():
            # Raw embedding for EMA+KL (surprise needs unbounded variance)
            embedding_raw = self._model.encode(
                text,
                normalize_embeddings=False,
                show_progress_bar=False,
            ).astype(np.float32)

            # Normalized embedding for raw features output (encoding model)
            norm = np.linalg.norm(embedding_raw)
            embedding_norm = embedding_raw / (norm + 1e-8) if norm > 1e-8 else embedding_raw

            self._prev_embedding_raw = embedding_raw
            self._prev_embedding_norm = embedding_norm
        else:
            if self._prev_embedding_raw is None:
                dim = self._model.get_sentence_embedding_dimension()
                embedding_raw = np.zeros(dim, dtype=np.float32)
                embedding_norm = np.zeros(dim, dtype=np.float32)
                self._prev_embedding_raw = embedding_raw
                self._prev_embedding_norm = embedding_norm
            else:
                embedding_raw = self._prev_embedding_raw
                embedding_norm = self._prev_embedding_norm

        return {
            "narrative": embedding_raw,          # → EMA+KL (surprise/uncertainty)
            "narrative_norm": embedding_norm,     # → raw features output (df2)
        }

    def reset(self) -> None:
        """Reset stored embedding. Call between films."""
        self._prev_embedding_raw = None
        self._prev_embedding_norm = None

