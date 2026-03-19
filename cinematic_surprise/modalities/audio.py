"""
modalities/audio.py

Audio feature extractor for three channels.

Each channel maps to a different level of the auditory cortical hierarchy:

    audio_mel    → A1 (primary auditory cortex, tonotopic map)
    audio_spec   → Belt cortex (secondary auditory cortex)
    audio_onset  → Parabelt / STS (transient/event detection)

All three are extracted from one 1-second audio segment per call.
No within-second aggregation — the 1-second window IS the observation unit.

IN:  numpy array of ~22050 float32 samples (1 second at 22050 Hz)
OUT: dict with three feature vectors
     "audio_mel"   → (128,) float32
     "audio_spec"  → (6,)   float32
     "audio_onset" → (1,)   float32
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from cinematic_surprise.config import (
    AUDIO_HOP_LENGTH,
    AUDIO_N_FFT,
    AUDIO_N_MELS,
    AUDIO_SAMPLE_RATE,
)

log = logging.getLogger(__name__)

try:
    import librosa
    _LIBROSA_OK = True
except ImportError:
    _LIBROSA_OK = False
    log.warning("librosa not installed. Audio modality will be unavailable.")


def extract_audio_features(
    segment: np.ndarray,
    sr:      int = AUDIO_SAMPLE_RATE,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Extract three-level audio features from a 1-second segment.

    Args:
        segment : float32 array of ~sr samples
        sr      : sample rate in Hz

    Returns:
        Dict with keys "audio_mel", "audio_spec", "audio_onset",
        or None if librosa is unavailable.

    Feature details
    ---------------
    audio_mel (128-d):
        Mean log mel-spectrogram across STFT time frames.
        128 triangular filters on the mel scale (mimicking cochlear spacing).
        Log-compressed. Captures the frequency energy profile of the second.
        Brain analog: A1 tonotopic organisation.

    audio_spec (6-d):
        [centroid_mean, centroid_std, rolloff_mean, rolloff_std, rms_mean, rms_std]
        Spectral centroid: centre of mass of the spectrum (brightness).
        Spectral rolloff: frequency below which 85% of energy sits.
        RMS energy: loudness.
        All normalised to [0, 1] scale.
        Brain analog: auditory belt cortex (timbral / roughness processing).

    audio_onset (1-d):
        Maximum onset strength in the window.
        Onset = sudden increase in spectral energy (drum hit, speech start).
        Brain analog: auditory parabelt, STS (temporal event detection).
    """
    if not _LIBROSA_OK:
        return None

    segment = np.asarray(segment, dtype=np.float32)

    # ── audio_mel: tonotopic frequency energy (A1) ────────────────────────
    mel = librosa.feature.melspectrogram(
        y=segment, sr=sr,
        n_mels=AUDIO_N_MELS,
        n_fft=AUDIO_N_FFT,
        hop_length=AUDIO_HOP_LENGTH,
    )
    mel_db   = librosa.power_to_db(mel + 1e-10, ref=1.0)
    mel_mean = mel_db.mean(axis=1).astype(np.float32)   # (128,)

    # ── audio_spec: timbral features (belt cortex) ────────────────────────
    max_freq = float(sr / 2)

    centroid = librosa.feature.spectral_centroid(
        y=segment, sr=sr, n_fft=AUDIO_N_FFT, hop_length=AUDIO_HOP_LENGTH
    ).ravel()

    rolloff = librosa.feature.spectral_rolloff(
        y=segment, sr=sr, n_fft=AUDIO_N_FFT, hop_length=AUDIO_HOP_LENGTH
    ).ravel()

    rms = librosa.feature.rms(
        y=segment, frame_length=AUDIO_N_FFT, hop_length=AUDIO_HOP_LENGTH
    ).ravel()

    spec = np.array([
        centroid.mean() / max_freq,
        centroid.std()  / max_freq,
        rolloff.mean()  / max_freq,
        rolloff.std()   / max_freq,
        rms.mean(),
        rms.std(),
    ], dtype=np.float32)   # (6,)

    # ── audio_onset: acoustic event detection (parabelt) ─────────────────
    onset_env = librosa.onset.onset_strength(
        y=segment, sr=sr, hop_length=AUDIO_HOP_LENGTH
    )
    onset_max = np.array([onset_env.max()], dtype=np.float32)   # (1,)

    return {
        "audio_mel":   mel_mean,
        "audio_spec":  spec,
        "audio_onset": onset_max,
    }
