"""
Stage 3 — Dataset Augmentation

Writes synthetic spectrum caption files alongside original HumanML3D captions.
Original captions are NEVER modified or removed.

Output per motion:
    text_dir/<motion_id>.txt          ← original (unchanged)
    text_dir/<motion_id>_spec_0.txt   ← base spectrum caption
    text_dir/<motion_id>_spec_1.txt   ← midpoint spectrum caption
    text_dir/<motion_id>_spec_2.txt   ← variant spectrum caption

Each synthetic .txt follows the HumanML3D format:
    <caption>#<token1/POS token2/POS ...>#0.0#0.0
(f_tag=0.0, to_tag=0.0 means "full sequence")
"""

from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .analyzer import SpectrumAnalyzer
from .synthesizer import SpectrumCaptionSynthesizer


# Minimal POS-tagged tokenizer for spectrum captions.
# Real HumanML3D tokens use GloVe-compatible POS tags, but the spectrum prefix
# tokens are novel structured strings — we treat them as "OTHER".
def _tokenize_caption(caption: str) -> str:
    """Very light tokenizer: lowercases, splits on space, tags everything OTHER.
    Only strip trailing sentence punctuation, not decimal points inside tags."""
    text = caption.lower().rstrip(".,").replace(",", "")
    tokens = text.split()
    return " ".join(f"{t}/OTHER" for t in tokens)


def _fmt_line(caption: str) -> str:
    tokens = _tokenize_caption(caption)
    return f"{caption}#{tokens}#0.0#0.0\n"


def _perturb_scores(
    scores: Dict[str, float],
    noise_std: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Add small Gaussian noise to scores, clamping to [0, 1]."""
    rng = rng or np.random.default_rng()
    return {
        k: float(np.clip(v + rng.normal(0, noise_std), 0.0, 1.0))
        for k, v in scores.items()
    }


def _midpoint_scores(
    scores: Dict[str, float],
    top_n: int = 2,
) -> Dict[str, float]:
    """Pull the top-2 dimensions toward each other (simulate partial blend)."""
    sorted_dims = sorted(scores, key=lambda d: scores[d], reverse=True)
    result = dict(scores)
    if len(sorted_dims) >= 2:
        d1, d2 = sorted_dims[0], sorted_dims[1]
        mid = (result[d1] + result[d2]) / 2.0
        result[d1] = mid + 0.1
        result[d2] = mid - 0.1
        for k in result:
            result[k] = float(np.clip(result[k], 0.0, 1.0))
    return result


class DatasetAugmenter:
    """
    Augments a HumanML3D-format dataset with synthetic spectrum captions.

    Parameters
    ----------
    motion_dir : str or Path
        Directory containing .npy motion files.
    text_dir : str or Path
        Directory containing original .txt caption files.
    spectrum_cache : str or Path, optional
        Path to pre-computed spectrum JSON (from SpectrumAnalyzer.analyze_directory).
        If None, spectra are computed on the fly.
    analyzer : SpectrumAnalyzer, optional
        Used when spectrum_cache is None.
    synthesizer : SpectrumCaptionSynthesizer, optional
        Defaults to a standard synthesizer.
    n_synthetic : int
        Number of synthetic caption files per motion (default 3).
    noise_std : float
        Score perturbation magnitude for variant captions.
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        motion_dir: str | Path,
        text_dir: str | Path,
        spectrum_cache: Optional[str | Path] = None,
        analyzer: Optional[SpectrumAnalyzer] = None,
        synthesizer: Optional[SpectrumCaptionSynthesizer] = None,
        n_synthetic: int = 3,
        noise_std: float = 0.05,
        seed: int = 42,
    ):
        self.motion_dir = Path(motion_dir)
        self.text_dir = Path(text_dir)
        self.n_synthetic = n_synthetic
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        self.synthesizer = synthesizer or SpectrumCaptionSynthesizer()
        self.analyzer = analyzer or SpectrumAnalyzer()

        if spectrum_cache is not None:
            with open(spectrum_cache) as f:
                self.spectrum_cache: Dict[str, Dict[str, float]] = json.load(f)
        else:
            self.spectrum_cache = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        split_file: Optional[str | Path] = None,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> int:
        """
        Augment all motions (or only those in split_file).

        Returns the number of motions successfully augmented.
        """
        if split_file is not None:
            ids = Path(split_file).read_text().strip().splitlines()
        else:
            ids = [p.stem for p in sorted(self.motion_dir.glob("*.npy"))]

        success = 0
        for motion_id in ids:
            try:
                self._augment_one(motion_id, overwrite=overwrite)
                success += 1
            except Exception as e:
                if verbose:
                    print(f"  SKIP {motion_id}: {e}")
        if verbose:
            print(f"Augmented {success}/{len(ids)} motions.")
        return success

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_scores(self, motion_id: str) -> Dict[str, float]:
        if motion_id in self.spectrum_cache:
            return self.spectrum_cache[motion_id]
        npy = self.motion_dir / f"{motion_id}.npy"
        scores = self.analyzer.analyze_file(npy)
        self.spectrum_cache[motion_id] = scores
        return scores

    def _augment_one(self, motion_id: str, overwrite: bool) -> None:
        scores = self._get_scores(motion_id)

        # Caption variants to write
        variants: List[Dict[str, float]] = [scores]
        variants.append(_midpoint_scores(scores))
        for _ in range(self.n_synthetic - 2):
            variants.append(_perturb_scores(scores, self.noise_std, self.rng))

        for i, variant_scores in enumerate(variants[: self.n_synthetic]):
            out_path = self.text_dir / f"{motion_id}_spec_{i}.txt"
            if out_path.exists() and not overwrite:
                continue
            caption = self.synthesizer.synthesize(variant_scores)
            out_path.write_text(_fmt_line(caption))


def augment_dataset(
    motion_dir: str | Path,
    text_dir: str | Path,
    spectrum_cache_path: Optional[str | Path] = None,
    calibration_path: Optional[str | Path] = None,
    split_file: Optional[str | Path] = None,
    n_synthetic: int = 3,
    overwrite: bool = False,
    seed: int = 42,
) -> int:
    """
    Convenience entry-point for the augmentation step.

    Example usage:
        python -c "from semantic_spectrum import augment_dataset; augment_dataset(...)"
    """
    if calibration_path is not None:
        analyzer = SpectrumAnalyzer.load_calibration(calibration_path)
    else:
        analyzer = SpectrumAnalyzer()

    aug = DatasetAugmenter(
        motion_dir=motion_dir,
        text_dir=text_dir,
        spectrum_cache=spectrum_cache_path,
        analyzer=analyzer,
        n_synthetic=n_synthetic,
        seed=seed,
    )
    return aug.run(split_file=split_file, overwrite=overwrite)
