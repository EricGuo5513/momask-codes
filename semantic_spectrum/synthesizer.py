"""
Spectrum Caption Synthesizer

Converts a spectrum score dict into a structured prompt string for conditioning
the MaskTransformer and ResidualTransformer.

Output format:
    [dim1:0.82][dim2:0.18][dim3:0.22] A person <verb phrase>.

The key function for the intensity-slider pipeline is synthesize_interpolated(),
which produces a sequence of captions sweeping smoothly between two points in
spectrum space — e.g. from a source walking motion toward a dance target.
These captions map directly onto the intensity levels used in the user study
(15%, 30%, 50%, 75%, 95% re-rendering intensity).

DESIGN INVARIANTS:
  - Dimension ordering is fixed (alphabetical) so token positions are stable.
  - Only dimensions scoring >= tag_threshold appear in the prefix.
  - Small score changes -> small string changes -> smooth shift in CLIP embedding.
  - The natural language tail is intentionally minimal and deterministic.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

from .dimensions import DIMENSIONS


# Verb templates per dominant dimension pair (primary, secondary)
# Kept intentionally sparse — linguistic diversity is NOT a goal here.
_VERB_TEMPLATES: Dict[Tuple[str, str], str] = {
    ("walk",  "dance"): "walks forward with rhythmic upper body movement",
    ("walk",  "run"):   "walks at a brisk pace",
    ("walk",  "spin"):  "walks while turning slightly",
    ("walk",  "wave"):  "walks and waves",
    ("dance", "walk"):  "dances with a walking base",
    ("dance", "spin"):  "dances and spins",
    ("dance", "jump"):  "dances with jumping",
    ("run",   "jump"):  "runs and leaps",
    ("run",   "walk"):  "jogs forward",
    ("jump",  "spin"):  "jumps and spins",
    ("kick",  "walk"):  "walks and kicks",
    ("spin",  "dance"): "spins with expressive movement",
    ("wave",  "stand"): "stands and waves",
    ("stand", "wave"):  "stands gesturing with arms",
}

_SINGLE_VERB: Dict[str, str] = {
    "walk":  "walks forward",
    "dance": "dances expressively",
    "run":   "runs forward",
    "jump":  "jumps",
    "spin":  "spins",
    "kick":  "kicks",
    "wave":  "waves",
    "stand": "stands still",
}


class SpectrumCaptionSynthesizer:
    """
    Converts semantic spectrum scores -> structured prompt string.

    Parameters
    ----------
    tag_threshold : float
        Dimensions below this score are omitted from the tag prefix.
    score_decimals : int
        Number of decimal places in score tags (e.g. 2 → [walk:0.82]).
    max_tags : int
        Maximum tags shown in the prefix (top-N by score).
    dimension_order : list of str, optional
        Fixed ordering of dimensions in the prefix.  Defaults to sorted(DIMENSIONS).
    """

    def __init__(
        self,
        tag_threshold: float = 0.05,
        score_decimals: int = 2,
        max_tags: int = 4,
        dimension_order: Optional[List[str]] = None,
    ):
        self.tag_threshold = tag_threshold
        self.score_decimals = score_decimals
        self.max_tags = max_tags
        self.dimension_order = dimension_order or sorted(DIMENSIONS.keys())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(self, scores: Dict[str, float]) -> str:
        """
        Parameters
        ----------
        scores : dict
            Output of SpectrumAnalyzer.analyze().

        Returns
        -------
        str  — e.g. "[walk:0.82][dance:0.18] A person walks forward with rhythmic upper body movement."
        """
        prefix = self._build_prefix(scores)
        verb = self._select_verb(scores)
        return f"{prefix} A person {verb}."

    def synthesize_interpolated(
        self,
        scores_a: Dict[str, float],
        scores_b: Dict[str, float],
        steps: int = 5,
    ) -> List[str]:
        """
        Produce a sequence of captions interpolating linearly between two spectrum
        score dicts.  This is the core function for the intensity-slider pipeline.

        Parameters
        ----------
        scores_a : dict
            Source spectrum scores (e.g. from a walking input video via MediaPipe).
        scores_b : dict
            Target spectrum scores (e.g. a full-dance point in spectrum space).
        steps : int
            Number of interpolation steps including both endpoints.
            Default 5 maps onto the user study intensity levels (0%, 25%, 50%, 75%, 100%).
            Use 7 for finer control (0%, ~17%, ~33%, 50%, ~67%, ~83%, 100%).

        Returns
        -------
        list of str
            Ordered captions from scores_a (step 0) to scores_b (step N-1).
            Feed each caption to the trained MaskTransformer to generate the
            corresponding motion at that intensity level.

        Example
        -------
            source = analyzer.analyze(walking_motion)   # from MediaPipe input
            target = {"dance": 0.9, "walk": 0.1, ...}  # desired end point
            captions = synthesizer.synthesize_interpolated(source, target, steps=5)
            # -> 5 captions, each progressively more dance-like
        """
        captions = []
        for i in range(steps):
            t = i / max(steps - 1, 1)
            blended = {
                k: (1 - t) * scores_a.get(k, 0.0) + t * scores_b.get(k, 0.0)
                for k in set(scores_a) | set(scores_b)
            }
            captions.append(self.synthesize(blended))
        return captions

    def format_tag(self, dim: str, score: float) -> str:
        fmt = f"[{dim}:{score:.{self.score_decimals}f}]"
        return fmt

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_prefix(self, scores: Dict[str, float]) -> str:
        # Filter + sort by score descending, but output in fixed dimension_order
        active = {
            d: scores.get(d, 0.0)
            for d in self.dimension_order
            if scores.get(d, 0.0) >= self.tag_threshold
        }
        # Keep top-N by score for brevity
        top_dims = sorted(active, key=lambda d: active[d], reverse=True)[: self.max_tags]
        # Re-order by fixed dimension_order for stable token positions
        ordered = [d for d in self.dimension_order if d in top_dims]
        return "".join(self.format_tag(d, active[d]) for d in ordered)

    def _select_verb(self, scores: Dict[str, float]) -> str:
        sorted_dims = sorted(scores, key=lambda d: scores[d], reverse=True)
        if not sorted_dims:
            return "moves"

        primary = sorted_dims[0]
        if scores[primary] < self.tag_threshold:
            return "moves"

        secondary = sorted_dims[1] if len(sorted_dims) > 1 else None

        if secondary and scores[secondary] >= self.tag_threshold:
            key = (primary, secondary)
            if key in _VERB_TEMPLATES:
                return _VERB_TEMPLATES[key]

        return _SINGLE_VERB.get(primary, "moves")
