"""
modalities/semantic.py

CLIP image encoder — semantic level.

CLIP (Radford et al. 2021) was trained on 400M web image-text pairs.
Its embedding space captures semantic content beyond the 1000 ImageNet
categories: mood, atmosphere, artistic style, abstract concepts.

Brain analog: prefrontal cortex (DLPFC, VLPFC), anterior temporal lobe,
              angular gyrus.

IN:  batch of BGR frames (B, H, W, 3) uint8
OUT: dict {"semantic": (B, 512)} float32 numpy array
     (768 if ViT-L/14 is selected in config)
"""

from __future__ import annotations

import logging
from typing import Dict

import cv2
import numpy as np
import torch

from cinematic_surprise.config import CLIP_DEVICE, CLIP_MODEL, CNN_INPUT_SIZE

log = logging.getLogger(__name__)

try:
    import clip
    _CLIP_OK = True
except ImportError:
    _CLIP_OK = False
    log.warning(
        "openai-clip not installed. Semantic modality will be unavailable. "
        "Install with: pip install git+https://github.com/openai/CLIP.git"
    )


class SemanticExtractor:
    """
    CLIP image encoder with batched inference.

    Args:
        model_name : CLIP model variant. Default from config (ViT-B/32).
        device     : 'cuda' or 'cpu'. Falls back to cpu if cuda unavailable.
    """

    def __init__(
        self,
        model_name: str = CLIP_MODEL,
        device:     str = CLIP_DEVICE,
    ):
        self.available = _CLIP_OK
        if not _CLIP_OK:
            return

        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )

        log.info(f"Loading CLIP model '{model_name}'...")
        self._model, self._preprocess_fn = clip.load(
            model_name, device=self.device
        )
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

        log.info(
            f"CLIP '{model_name}' loaded on {self.device}. "
            f"Embedding dim: {self._model.visual.output_dim}"
        )

    @torch.no_grad()
    def extract(self, frames_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract CLIP embeddings for a batch of frames.

        Args:
            frames_bgr : (B, H, W, 3) uint8 BGR batch

        Returns:
            {"semantic": (B, 512)} float32  (or empty dict if unavailable)
        """
        if not self.available:
            return {}

        # CLIP preprocessing: BGR → PIL-like tensor via torchvision transforms
        import torch
        from PIL import Image

        images = []
        for frame in frames_bgr:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            images.append(self._preprocess_fn(pil))

        batch = torch.stack(images).to(self.device)
        feats = self._model.encode_image(batch)

        # L2-normalise: CLIP embeddings are used as cosine-space vectors
        feats = feats / feats.norm(dim=-1, keepdim=True)

        return {"semantic": feats.cpu().numpy().astype(np.float32)}
