"""
pipeline.py — CinematicSurprisePipeline

Main entry point. Orchestrates Phase 2 of the pipeline:
    - Read video frame by frame (1-second windows)
    - Extract features per modality
    - Run EMA + KL + uncertainty per channel per frame/second
    - Aggregate to per-second scalars
    - Run post-hoc aggregation (interactions, combined columns)
    - Return pandas DataFrame with 48 columns

Phase 1 (Whisper transcription) is handled separately:
    from cinematic_surprise.io.transcript import transcribe
    transcribe('film.mp4', output='film_transcript.csv')

Usage
-----
    from cinematic_surprise import CinematicSurprisePipeline

    pipe = CinematicSurprisePipeline()
    df   = pipe.run('film.mp4', transcript='film_transcript.csv')
    df.to_parquet('film_surprise.parquet', index=False)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from cinematic_surprise import config as cfg
from cinematic_surprise.io.audio import AudioExtractor
from cinematic_surprise.io.transcript import TranscriptReader
from cinematic_surprise.io.video import VideoReader
from cinematic_surprise.modalities.audio import extract_audio_features
from cinematic_surprise.modalities.face import FaceExtractor
from cinematic_surprise.modalities.motion import MotionExtractor
from cinematic_surprise.modalities.narrative import NarrativeExtractor
from cinematic_surprise.modalities.semantic import SemanticExtractor
from cinematic_surprise.modalities.visual import VisualExtractor
from cinematic_surprise.uncertainty_and_surprise.aggregator import run_all
from cinematic_surprise.uncertainty_and_surprise.estimator import OnlineGaussianEstimator

log = logging.getLogger(__name__)

# Channels that produce frame-level observations (need within-second mean)
_FRAME_LEVEL_CHANNELS = {"L1", "L2", "L3", "L4", "semantic", "motion", "emotion", "faces"}

# Channels that produce one observation per second (no within-second aggregation)
_SECOND_LEVEL_CHANNELS = {"audio_mel", "audio_spec", "audio_onset", "narrative"}


class CinematicSurprisePipeline:
    """
    Per-second Bayesian surprise and uncertainty estimator for film stimuli.

    All parameters default to config.py values and can be overridden.

    Args:
        alpha       : Dict of per-channel EMA learning rates.
                      Defaults to config.ALPHA.
        output_fmt  : 'parquet' or 'csv'. Used by save().
        max_seconds : Process only the first N seconds (None = whole film).
        batch_size  : CNN frames per forward pass.
        cut_threshold: Chi-squared scene-cut threshold.
    """

    def __init__(
        self,
        alpha:          Optional[Dict[str, float]] = None,
        output_fmt:     str = "parquet",
        max_seconds:    Optional[int] = None,
        batch_size:     int = cfg.BATCH_SIZE,
        cut_threshold:  float = 0.15,
    ):
        self.output_fmt    = output_fmt
        self.max_seconds   = max_seconds
        self.batch_size    = batch_size
        self.cut_threshold = cut_threshold

        # Shared estimator: one EMA+KL belief per named channel
        self.estimator = OnlineGaussianEstimator(alpha=alpha)

        # Modality extractors (loaded once, reused across films)
        log.info("Loading visual extractor (ResNet-50)...")
        self.visual    = VisualExtractor(device=cfg.RESNET_DEVICE)

        log.info(f"Loading semantic extractor (CLIP {cfg.CLIP_MODEL})...")
        self.semantic  = SemanticExtractor(
            model_name=cfg.CLIP_MODEL,
            device=cfg.CLIP_DEVICE,
        )

        self.motion    = MotionExtractor()
        self.face      = FaceExtractor()
        self.narrative = NarrativeExtractor()

        log.info("Pipeline ready.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        video_path:  str | Path,
        transcript:  Optional[str | Path] = None,
    ) -> pd.DataFrame:
        """
        Process one film and return a per-second DataFrame.

        Args:
            video_path : Path to video file (MP4, MKV, AVI, ...)
            transcript : Path to standardised transcript CSV.
                         If None, narrative channel is skipped.

        Returns:
            pandas DataFrame with 48 columns (9 metadata + 36 channels + 3 aggregates)
        """
        video_path = Path(video_path)

        # Reset all beliefs and stateful extractors for fresh film
        self._reset(video_path)

        # Load audio and transcript
        log.info("Loading audio...")
        audio_ext = AudioExtractor(str(video_path), sr=cfg.AUDIO_SAMPLE_RATE)

        transcript_reader: Optional[TranscriptReader] = None
        if transcript is not None:
            log.info(f"Loading transcript from '{transcript}'...")
            transcript_reader = TranscriptReader(transcript)
        else:
            log.info("No transcript provided. Narrative channel will be skipped.")

        # Main loop
        rows = []
        with VideoReader(
            video_path,
            max_seconds=self.max_seconds,
            cut_threshold=self.cut_threshold,
        ) as vr:
            total_s = min(vr.n_seconds, self.max_seconds or vr.n_seconds)

            for second_idx, frames, has_cut in tqdm(
                vr.iter_seconds(),
                total=total_s,
                desc=f"Processing '{video_path.name}'",
                unit="s",
            ):
                row = self._process_second(
                    second_idx=second_idx,
                    frames=frames,
                    has_cut=has_cut,
                    audio_ext=audio_ext,
                    transcript_reader=transcript_reader,
                )
                rows.append(row)

        df = pd.DataFrame(rows)

        # Post-hoc: compute interaction columns and aggregate columns
        log.info("Computing interactions and aggregates...")
        df = run_all(df)

        log.info(
            f"Done. {len(df)} seconds processed. "
            f"Columns: {len(df.columns)}. "
            f"Peak surprise_combined at t={df['surprise_combined'].idxmax()}s."
        )
        return df

    def save(self, df: pd.DataFrame, path: str | Path) -> Path:
        """
        Save the output DataFrame to disk.

        Args:
            df   : Output of run()
            path : Output path (extension added automatically if not present)

        Returns:
            Actual path written.
        """
        path = Path(path)
        if self.output_fmt == "parquet":
            out = path.with_suffix(".parquet")
            df.to_parquet(out, index=False)
        else:
            out = path.with_suffix(".csv")
            df.to_csv(out, index=False)
        log.info(f"Saved → '{out}'")
        return out

    # ── Internal ───────────────────────────────────────────────────────────────

    def _reset(self, video_path: Path) -> None:
        """Reset all stateful components for a new film."""
        self.estimator.reset()
        self.motion.reset()
        self.narrative.reset()
        log.info(f"State reset for '{video_path.name}'")

    def _process_second(
        self,
        second_idx:        int,
        frames:            List[np.ndarray],
        has_cut:           bool,
        audio_ext:         AudioExtractor,
        transcript_reader: Optional[TranscriptReader],
    ) -> dict:
        """
        Process one 1-second window → one row dict.

        Frame-level channels (L1-L4, semantic, motion, emotion, faces):
            EMA+KL per frame → collect frame-level surprise and uncertainty
            Per-second value = mean of frame-level values

        Second-level channels (audio_mel, audio_spec, audio_onset, narrative):
            One feature vector per second → one EMA+KL update per second
        """
        frames_arr = np.stack(frames, axis=0)   # (B, H, W, 3)
        B = len(frames_arr)

        row: dict = {
            "time_s":    second_idx,
            "n_frames":  B,
            "scene_cut": has_cut,
        }

        # ── Frame-level: visual (L1–L4) ────────────────────────────────────
        # Process in sub-batches for GPU memory safety
        visual_feats:   Dict[str, List[np.ndarray]] = {ch: [] for ch in ["L1", "L2", "L3", "L4"]}
        semantic_feats: List[np.ndarray] = []
        motion_feats:   List[np.ndarray] = []

        for start in range(0, B, self.batch_size):
            batch = frames_arr[start: start + self.batch_size]

            # ResNet L1-L4
            vis = self.visual.extract(batch)
            for ch in ["L1", "L2", "L3", "L4"]:
                visual_feats[ch].append(vis[ch])

            # CLIP semantic
            if self.semantic.available:
                sem = self.semantic.extract(batch)
                if "semantic" in sem:
                    semantic_feats.append(sem["semantic"])

        # Concatenate sub-batches → (B, d) per channel
        for ch in ["L1", "L2", "L3", "L4"]:
            feat_seq = np.concatenate(visual_feats[ch], axis=0)   # (B, d)
            s_vals, u_vals = zip(*[
                self.estimator.update(feat_seq[i], ch)
                for i in range(B)
            ])
            row[f"surprise_{ch}"]    = float(np.mean(s_vals))
            row[f"uncertainty_{ch}"] = float(np.mean(u_vals))

        if semantic_feats:
            sem_seq = np.concatenate(semantic_feats, axis=0)   # (B, d)
            s_vals, u_vals = zip(*[
                self.estimator.update(sem_seq[i], "semantic")
                for i in range(len(sem_seq))
            ])
            row["surprise_semantic"]    = float(np.mean(s_vals))
            row["uncertainty_semantic"] = float(np.mean(u_vals))
        else:
            row["surprise_semantic"]    = np.nan
            row["uncertainty_semantic"] = np.nan

        # ── Frame-level: motion ────────────────────────────────────────────
        motion_s, motion_u = [], []
        for frame in frames_arr:
            feat = self.motion.extract(frame)
            s, u = self.estimator.update(feat, "motion")
            motion_s.append(s)
            motion_u.append(u)
        row["surprise_motion"]    = float(np.mean(motion_s))
        row["uncertainty_motion"] = float(np.mean(motion_u))

        # ── Frame-level: face / emotion ────────────────────────────────────
        face_result = self.face.extract(list(frames_arr))

        for face_ch in ["emotion", "faces"]:
            feat = face_result[face_ch]
            # Face features are already aggregated to per-second vectors
            # by FaceExtractor, so we do one EMA+KL update per second
            s, u = self.estimator.update(feat, face_ch)
            row[f"surprise_{face_ch}"]    = s
            row[f"uncertainty_{face_ch}"] = u

        # Face metadata columns (not fed into EMA+KL)
        row["n_faces_mean"]          = face_result["n_faces_mean"]
        row["n_faces_std"]           = face_result["n_faces_std"]
        row["face_coverage_mean"]    = face_result["face_coverage_mean"]
        row["face_coverage_std"]     = face_result["face_coverage_std"]
        row["dominant_emotion_idx"]  = face_result["dominant_emotion_idx"]
        row["dominant_emotion"]      = face_result["dominant_emotion"]

        # ── Second-level: audio ────────────────────────────────────────────
        if audio_ext.available:
            segment = audio_ext.get_segment(second_idx)
            if segment is not None:
                audio_feats = extract_audio_features(segment, sr=audio_ext.sr)
                if audio_feats:
                    for audio_ch in ["audio_mel", "audio_spec", "audio_onset"]:
                        s, u = self.estimator.update(audio_feats[audio_ch], audio_ch)
                        row[f"surprise_{audio_ch}"]    = s
                        row[f"uncertainty_{audio_ch}"] = u
                else:
                    for audio_ch in ["audio_mel", "audio_spec", "audio_onset"]:
                        row[f"surprise_{audio_ch}"]    = np.nan
                        row[f"uncertainty_{audio_ch}"] = np.nan
            else:
                for audio_ch in ["audio_mel", "audio_spec", "audio_onset"]:
                    row[f"surprise_{audio_ch}"]    = np.nan
                    row[f"uncertainty_{audio_ch}"] = np.nan
        else:
            for audio_ch in ["audio_mel", "audio_spec", "audio_onset"]:
                row[f"surprise_{audio_ch}"]    = np.nan
                row[f"uncertainty_{audio_ch}"] = np.nan

        # ── Second-level: narrative ────────────────────────────────────────
        if transcript_reader is not None:
            text = transcript_reader.get_words(second_idx)
            nar_feat = self.narrative.extract(text)
            if "narrative" in nar_feat:
                s, u = self.estimator.update(nar_feat["narrative"], "narrative")
                row["surprise_narrative"]    = s
                row["uncertainty_narrative"] = u
            else:
                row["surprise_narrative"]    = np.nan
                row["uncertainty_narrative"] = np.nan
        else:
            row["surprise_narrative"]    = np.nan
            row["uncertainty_narrative"] = np.nan

        return row
