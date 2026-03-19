"""
modalities/motion.py

Dense optical flow feature extractor — motion level.

Algorithm: Farneback polynomial expansion (cv2.calcOpticalFlowFarneback).
Brain analog: middle temporal area (MT/V5), MST, superior temporal sulcus.

Feature vector (19-d):
    - 16-bin magnitude histogram (motion energy distribution)
    - mean magnitude / FLOW_MAX_MAG  (normalised mean speed)
    - mean sin(angle)                (mean motion direction sin)
    - mean cos(angle)                (mean motion direction cos)

IN:  consecutive BGR frames, one at a time
OUT: (19,) float32 feature vector per frame pair

Note: requires a previous frame to compute flow.
Returns zeros on the first frame (no previous frame available).
Call reset() between films.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from cinematic_surprise.config import (
    FLOW_MAX_MAG,
    FLOW_N_BINS,
    FLOW_RESIZE,
)


class MotionExtractor:
    """
    Stateful optical flow extractor.

    Maintains the previous frame internally. Call extract(frame) per frame;
    returns a (19,) feature vector. Returns zeros on the first call.

    Args:
        resize : (W, H) tuple for flow computation resolution.
                 Smaller = faster. Default (320, 180) from config.
    """

    def __init__(self, resize: tuple = FLOW_RESIZE):
        self.resize = resize
        self._prev_gray: Optional[np.ndarray] = None

    def extract(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Compute optical flow from previous frame to this frame.

        Args:
            frame_bgr : (H, W, 3) uint8 BGR frame

        Returns:
            (19,) float32 feature vector.
            All zeros if no previous frame (first call after reset).
        """
        gray = cv2.cvtColor(
            cv2.resize(frame_bgr, self.resize, interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY
        )

        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return np.zeros(FLOW_N_BINS + 3, dtype=np.float32)

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        self._prev_gray = gray

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # 16-bin magnitude histogram (captures motion energy distribution)
        hist, _ = np.histogram(
            mag.ravel(),
            bins=FLOW_N_BINS,
            range=(0.0, FLOW_MAX_MAG),
            density=True,
        )

        # Summary statistics
        extra = np.array([
            mag.mean() / (FLOW_MAX_MAG + 1e-9),    # normalised mean speed
            np.mean(np.sin(ang)),                   # mean direction sin
            np.mean(np.cos(ang)),                   # mean direction cos
        ], dtype=np.float32)

        return np.concatenate([hist.astype(np.float32), extra])

    def reset(self) -> None:
        """Reset stored previous frame. Call between films."""
        self._prev_gray = None
