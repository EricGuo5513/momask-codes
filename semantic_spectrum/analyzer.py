"""
Motion Spectrum Analyzer

Extracts independent per-dimension semantic scores from raw joint coordinates.

Input format : (T, 22, 3) — Y-up, units in meters, 20 fps.
Output format : dict mapping dimension name -> float in [0, 1].

Scores are NOT softmax-normalised — a motion can score high on multiple
dimensions simultaneously (e.g. a dance with a running base gives both
dance and run non-zero scores).

Typical usage
-------------
    from semantic_spectrum.analyzer import SpectrumAnalyzer

    analyzer = SpectrumAnalyzer.load_calibration("semantic_spectrum_data/calibration.json")
    scores = analyzer.analyze(motion)          # motion: np.ndarray (T, 22, 3)
    # -> {"walk": 0.82, "dance": 0.31, "run": 0.05, ...}

Pre-computed scores for the full HumanML3D training set are available in
semantic_spectrum_data/spectrum_scores.json for browsing without re-running.
"""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .dimensions import DIMENSIONS


def _validate_joints(motion: np.ndarray, min_frames: int) -> None:
    if motion.ndim != 3 or motion.shape[1] != 22 or motion.shape[2] != 3:
        raise ValueError(
            f"Expected (T, 22, 3) raw joint array, got {motion.shape}. "
            "Each file should contain per-frame world-space joint positions."
        )
    if motion.shape[0] < min_frames:
        raise ValueError(
            f"Motion too short ({motion.shape[0]} frames < {min_frames} minimum)."
        )


class SpectrumAnalyzer:
    """
    Analyses a raw joint motion array and returns an independent semantic score
    per dimension in [0, 1].

    Parameters
    ----------
    dimensions : list of str, optional
        Which dimensions to analyse.  Defaults to all registered dimensions.
    calibration : dict[str, tuple[float, float]], optional
        Per-dimension (low, high) thresholds for linear rescaling to [0, 1].
        When None, extractor outputs (already clipped to [0, 1]) are used as-is.
    min_frames : int
        Sequences shorter than this are rejected.
    """

    def __init__(
        self,
        dimensions: Optional[List[str]] = None,
        calibration: Optional[Dict[str, Tuple[float, float]]] = None,
        min_frames: int = 20,
    ):
        self.dimensions = dimensions or list(DIMENSIONS.keys())
        for d in self.dimensions:
            if d not in DIMENSIONS:
                raise ValueError(f"Unknown dimension '{d}'. Available: {list(DIMENSIONS)}")
        self.calibration = calibration or {}
        self.min_frames = min_frames

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, motion: np.ndarray) -> Dict[str, float]:
        """
        Parameters
        ----------
        motion : np.ndarray, shape (T, 22, 3)
            Raw world-space joint positions in meters, Y-up, at 20 fps.

        Returns
        -------
        dict mapping dimension name -> score in [0, 1]
        """
        _validate_joints(motion, self.min_frames)

        scores: Dict[str, float] = {}
        for dim in self.dimensions:
            extractor, _ = DIMENSIONS[dim]
            raw = extractor(motion)
            scores[dim] = self._calibrate(dim, raw)
        return scores

    def analyze_file(self, path: str | Path) -> Dict[str, float]:
        motion = np.load(str(path))
        return self.analyze(motion)

    def analyze_directory(
        self,
        motion_dir: str | Path,
        output_json: Optional[str | Path] = None,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyse all .npy files in motion_dir.

        Returns dict mapping motion_id -> spectrum scores.
        Optionally writes results to output_json.
        """
        motion_dir = Path(motion_dir)
        files = sorted(motion_dir.glob("*.npy"))
        if verbose:
            print(f"Analysing {len(files)} motions in {motion_dir}")

        results: Dict[str, Dict[str, float]] = {}
        failed = 0
        for f in files:
            try:
                results[f.stem] = self.analyze_file(f)
            except Exception as e:
                if verbose:
                    print(f"  SKIP {f.name}: {e}")
                failed += 1

        if verbose:
            print(f"Done. {len(results)} succeeded, {failed} failed.")

        if output_json is not None:
            Path(output_json).write_text(json.dumps(results, indent=2))
            if verbose:
                print(f"Wrote spectrum cache -> {output_json}")

        return results

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def fit_calibration(
        self,
        motion_dir: str | Path,
        low_percentile: float = 5.0,
        high_percentile: float = 95.0,
        save_path: Optional[str | Path] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Run extractors over all motions and compute per-dimension percentile
        thresholds for linear rescaling to [0, 1].

        Call this once on your dataset before mass analysis.
        """
        motion_dir = Path(motion_dir)
        raw_scores: Dict[str, List[float]] = {d: [] for d in self.dimensions}

        for f in sorted(motion_dir.glob("*.npy")):
            try:
                motion = np.load(str(f))
                _validate_joints(motion, self.min_frames)
                for dim in self.dimensions:
                    extractor, _ = DIMENSIONS[dim]
                    raw_scores[dim].append(extractor(motion))
            except Exception:
                continue

        calibration: Dict[str, Tuple[float, float]] = {}
        for dim, vals in raw_scores.items():
            arr = np.array(vals)
            lo = float(np.percentile(arr, low_percentile))
            hi = float(np.percentile(arr, high_percentile))
            calibration[dim] = (lo, hi)
            print(f"  {dim:10s}: p{low_percentile:.0f}={lo:.4f}  p{high_percentile:.0f}={hi:.4f}")

        self.calibration = calibration

        if save_path is not None:
            Path(save_path).write_text(json.dumps(calibration, indent=2))
            print(f"Calibration saved -> {save_path}")

        return calibration

    @classmethod
    def load_calibration(cls, path: str | Path, **kwargs) -> "SpectrumAnalyzer":
        cal_raw = json.loads(Path(path).read_text())
        calibration = {k: tuple(v) for k, v in cal_raw.items()}
        return cls(calibration=calibration, **kwargs)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _calibrate(self, dim: str, raw: float) -> float:
        if dim not in self.calibration:
            return float(np.clip(raw, 0.0, 1.0))
        lo, hi = self.calibration[dim]
        if hi <= lo:
            return float(np.clip(raw, 0.0, 1.0))
        scaled = (raw - lo) / (hi - lo)
        return float(np.clip(scaled, 0.0, 1.0))
