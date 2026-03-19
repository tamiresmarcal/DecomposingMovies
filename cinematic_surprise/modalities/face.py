"""
modalities/face.py

Face and emotion feature extractor using DeepFace.

Two channels:
    emotion      (7-d) : coverage-weighted emotion probability vector
    faces        (3-d) : face count and coverage variability

Weighting logic:
    Each face's emotion vector is weighted by its bounding-box area as a
    percentage of total frame area. Larger face = closer to camera = more
    visually dominant = more weight. No arbitrary rule needed — the data
    decides automatically.

Zero-face handling:
    When no faces are detected, the emotion vector is all zeros.
    This is informationally meaningful: if previous seconds had faces,
    the EMA prior will have adapted to expect emotional signal; a sudden
    zero vector produces a surprise spike reflecting the disappearance
    of emotional content.

Two separate EMA+KL channels (not one):
    emotion      → FFA, amygdala, anterior STS
    faces        → OFA, posterior STS, fusiform gyrus
    These can dissociate in fMRI data: emotion changes without face count
    changing, or vice versa.

IN:  list of BGR frames for one second
OUT: dict with aggregated per-second vectors and metadata
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from cinematic_surprise.config import (
    DEEPFACE_BACKEND,
    EMOTION_LABELS,
    FACE_MIN_AREA_PCT,
    IDX_EMOTION,
)

log = logging.getLogger(__name__)

try:
    from deepface import DeepFace
    _DEEPFACE_OK = True
except ImportError:
    _DEEPFACE_OK = False
    log.warning(
        "deepface not installed. Face/emotion modality will be unavailable. "
        "Install with: pip install deepface"
    )


def _extract_frame(
    frame_bgr:  np.ndarray,
    backend:    str = DEEPFACE_BACKEND,
    min_area:   float = FACE_MIN_AREA_PCT,
) -> Tuple[np.ndarray, float, float]:
    """
    Extract coverage-weighted emotion vector from one frame.

    Returns:
        emotion_vec     : (7,) float32 — weighted emotion probabilities
        n_faces         : float — number of valid faces detected
        coverage_pct    : float — total face area as % of frame
    """
    if not _DEEPFACE_OK:
        return np.zeros(7, dtype=np.float32), 0.0, 0.0

    H, W = frame_bgr.shape[:2]
    frame_area = H * W

    try:
        results = DeepFace.analyze(
            frame_bgr,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend=backend,
            silent=True,
        )
    except Exception as e:
        log.debug(f"DeepFace failed on frame: {e}")
        return np.zeros(7, dtype=np.float32), 0.0, 0.0

    if not results:
        return np.zeros(7, dtype=np.float32), 0.0, 0.0

    emotion_vecs = []
    face_areas   = []

    for face in results:
        region = face.get("region", {})
        w = region.get("w", 0)
        h = region.get("h", 0)
        area = float(w * h)

        # Filter implausibly small detections (likely false positives)
        if area < frame_area * min_area:
            continue

        emo = face.get("emotion", {})
        vec = np.array(
            [emo.get(label, 0.0) for label in EMOTION_LABELS],
            dtype=np.float32,
        )
        # DeepFace returns percentages; normalise to sum to 1
        total = vec.sum()
        if total > 1e-9:
            vec /= total

        emotion_vecs.append(vec)
        face_areas.append(area)

    if not emotion_vecs:
        return np.zeros(7, dtype=np.float32), 0.0, 0.0

    areas   = np.array(face_areas)
    weights = areas / areas.sum()   # normalise: largest face gets most weight
    emotion_matrix = np.stack(emotion_vecs)

    weighted_emotion = (weights[:, None] * emotion_matrix).sum(axis=0)
    coverage_pct     = (areas.sum() / frame_area) * 100.0

    return weighted_emotion, float(len(emotion_vecs)), coverage_pct


class FaceExtractor:
    """
    Aggregates face/emotion features over a 1-second window.

    Call extract(frames) with the list of frames for one second.
    Returns feature vectors and metadata for that second.
    """

    def extract(self, frames: List[np.ndarray]) -> Dict:
        """
        Process all frames in a 1-second window.

        Per-frame emotion vectors are averaged across frames (unweighted —
        each frame contributes equally to the second-level representation).

        Args:
            frames : list of (H, W, 3) uint8 BGR frames

        Returns dict with:
            emotion_vec     : (7,) float32 → fed to EMA+KL as "emotion" channel
            faces_vec       : (3,) float32 → fed to EMA+KL as "faces" channel
                              [n_faces_mean, n_faces_std, face_coverage_std]
            n_faces_mean    : float  → metadata column
            n_faces_std     : float  → metadata column
            face_coverage_mean : float → metadata column
            face_coverage_std  : float → metadata column
            dominant_emotion_idx : int   → metadata column (-1 if no faces)
            dominant_emotion     : str   → metadata column
        """
        frame_emotions  = []
        frame_n_faces   = []
        frame_coverages = []

        for frame in frames:
            vec, n, cov = _extract_frame(frame)
            frame_emotions.append(vec)
            frame_n_faces.append(n)
            frame_coverages.append(cov)

        # Per-second aggregation: mean across frames
        emotion_vec    = np.stack(frame_emotions).mean(axis=0)   # (7,)
        n_faces_mean   = float(np.mean(frame_n_faces))
        n_faces_std    = float(np.std(frame_n_faces))
        coverage_mean  = float(np.mean(frame_coverages))
        coverage_std   = float(np.std(frame_coverages))

        # faces_vec: what goes into EMA+KL for the "faces" channel
        faces_vec = np.array(
            [n_faces_mean, n_faces_std, coverage_std],
            dtype=np.float32,
        )   # (3,)

        # Dominant emotion metadata
        if emotion_vec.sum() < 1e-6:
            dominant_idx   = -1
            dominant_label = "none"
        else:
            dominant_idx   = int(np.argmax(emotion_vec))
            dominant_label = IDX_EMOTION[dominant_idx]

        return {
            "emotion":              emotion_vec,       # → EMA+KL channel "emotion"
            "faces":                faces_vec,         # → EMA+KL channel "faces"
            "n_faces_mean":         n_faces_mean,      # metadata
            "n_faces_std":          n_faces_std,       # metadata
            "face_coverage_mean":   coverage_mean,     # metadata
            "face_coverage_std":    coverage_std,      # metadata
            "dominant_emotion_idx": dominant_idx,      # metadata
            "dominant_emotion":     dominant_label,    # metadata
        }
