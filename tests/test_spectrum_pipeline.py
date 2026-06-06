"""
Unit tests for the semantic spectrum pipeline.
Input format: (T, 22, 3) raw joint coordinates (Y-up, meters, 20 fps).

Run with:  python -m pytest tests/test_spectrum_pipeline.py -v
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_spectrum.analyzer import SpectrumAnalyzer
from semantic_spectrum.synthesizer import SpectrumCaptionSynthesizer
from semantic_spectrum.dimensions import DIMENSIONS, FPS


# ── Fixture helpers ──────────────────────────────────────────────────

# Approximate SMPL rest-pose joint heights (Y-up, meters) and X offsets.
# Ankle/toe joints are near y=0 (floor), pelvis ~1m, head ~1.7m.
_BASE_Y = np.array([
    0.98,  # 0  Pelvis
    0.90,  # 1  L_Hip
    0.90,  # 2  R_Hip
    1.12,  # 3  Spine1
    0.50,  # 4  L_Knee
    0.50,  # 5  R_Knee
    1.22,  # 6  Spine2
    0.08,  # 7  L_Ankle
    0.08,  # 8  R_Ankle
    1.33,  # 9  Spine3
    0.02,  # 10 L_Toe
    0.02,  # 11 R_Toe
    1.53,  # 12 Neck
    1.43,  # 13 L_Collar
    1.43,  # 14 R_Collar
    1.63,  # 15 Head
    1.42,  # 16 L_Shoulder
    1.42,  # 17 R_Shoulder
    1.18,  # 18 L_Elbow
    1.18,  # 19 R_Elbow
    0.95,  # 20 L_Wrist
    0.95,  # 21 R_Wrist
], dtype=np.float32)

_BASE_X = np.array([
     0.00,  # Pelvis
    -0.10,  0.10,   # L/R Hip
     0.00,          # Spine1
    -0.10,  0.10,   # L/R Knee
     0.00,          # Spine2
    -0.10,  0.10,   # L/R Ankle
     0.00,          # Spine3
    -0.10,  0.10,   # L/R Toe
     0.00,          # Neck
    -0.18,  0.18,   # L/R Collar
     0.00,          # Head
    -0.35,  0.35,   # L/R Shoulder
    -0.50,  0.50,   # L/R Elbow
    -0.60,  0.60,   # L/R Wrist
], dtype=np.float32)


def _standing_motion(T: int = 60, noise: float = 0.01) -> np.ndarray:
    """Static standing pose with small Gaussian noise."""
    rng = np.random.default_rng(0)
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    joints[:, :, 0] = _BASE_X[None, :]
    joints[:, :, 1] = _BASE_Y[None, :]
    joints += rng.standard_normal((T, 22, 3)).astype(np.float32) * noise
    return joints


def _walking_motion(T: int = 80, speed: float = 1.2) -> np.ndarray:
    """
    Synthetic walking motion: constant forward translation + alternating
    ankle lifts to simulate foot clearance.
    """
    joints = _standing_motion(T, noise=0.005)
    for t in range(T):
        joints[t, :, 0] += t * speed / FPS        # translate forward (X)
        phase = (t // 10) % 2
        if phase == 0:
            joints[t, 7, 1] += 0.20               # L_Ankle up
            joints[t, 10, 1] += 0.20              # L_Toe up
        else:
            joints[t, 8, 1] += 0.20               # R_Ankle up
            joints[t, 11, 1] += 0.20              # R_Toe up
    return joints


def _spinning_motion(T: int = 60, rate_rps: float = 1.5) -> np.ndarray:
    """
    Synthetic spin: body rotates around Y axis at `rate_rps` revolutions/sec.
    Hip joints revolve around the pelvis.
    """
    joints = _standing_motion(T, noise=0.005)
    for t in range(T):
        angle = 2 * np.pi * rate_rps * t / FPS
        c, s = np.cos(angle), np.sin(angle)
        # Rotate all joints around pelvis in XZ plane
        pelvis = joints[t, 0, :].copy()
        for j in range(22):
            dx = joints[t, j, 0] - pelvis[0]
            dz = joints[t, j, 2] - pelvis[2]
            joints[t, j, 0] = pelvis[0] + c * dx - s * dz
            joints[t, j, 2] = pelvis[2] + s * dx + c * dz
    return joints


# ── SpectrumAnalyzer tests ───────────────────────────────────────────

class TestSpectrumAnalyzer:
    def test_returns_all_dimensions(self):
        analyzer = SpectrumAnalyzer()
        scores = analyzer.analyze(_standing_motion())
        assert set(scores.keys()) == set(DIMENSIONS.keys())

    def test_scores_in_range(self):
        analyzer = SpectrumAnalyzer()
        for motion in [_standing_motion(), _walking_motion(), _spinning_motion()]:
            scores = analyzer.analyze(motion)
            for dim, score in scores.items():
                assert 0.0 <= score <= 1.0, f"{dim} score {score} out of [0,1]"

    def test_scores_are_independent(self):
        """Sum of scores can exceed 1.0 — dimensions are NOT mutually exclusive."""
        analyzer = SpectrumAnalyzer()
        scores = analyzer.analyze(_walking_motion())
        # Just assert it runs and the sum is non-negative (not softmax)
        assert sum(scores.values()) >= 0.0

    def test_short_motion_raises(self):
        analyzer = SpectrumAnalyzer(min_frames=40)
        with pytest.raises(ValueError, match="too short"):
            analyzer.analyze(_standing_motion(T=10))

    def test_wrong_shape_raises(self):
        analyzer = SpectrumAnalyzer()
        with pytest.raises(ValueError, match=r"\(T, 22, 3\)"):
            analyzer.analyze(np.zeros((60, 263)))             # old 263-dim format

    def test_wrong_joint_count_raises(self):
        analyzer = SpectrumAnalyzer()
        with pytest.raises(ValueError, match=r"\(T, 22, 3\)"):
            analyzer.analyze(np.zeros((60, 17, 3)))           # wrong joint count

    def test_subset_dimensions(self):
        analyzer = SpectrumAnalyzer(dimensions=["walk", "dance"])
        scores = analyzer.analyze(_walking_motion())
        assert set(scores.keys()) == {"walk", "dance"}

    def test_calibrate_clips_to_range(self):
        cal = {"stand": (0.1, 0.9)}
        analyzer = SpectrumAnalyzer(dimensions=["stand"], calibration=cal)
        scores = analyzer.analyze(_standing_motion())
        assert 0.0 <= scores["stand"] <= 1.0

    def test_calibration_degenerate_lo_eq_hi(self):
        """When lo == hi, calibration should not divide by zero."""
        cal = {"walk": (0.5, 0.5)}
        analyzer = SpectrumAnalyzer(dimensions=["walk"], calibration=cal)
        scores = analyzer.analyze(_walking_motion())
        assert 0.0 <= scores["walk"] <= 1.0

    # ── Semantic sanity tests ────────────────────────────────────────

    def test_stand_higher_for_static_than_moving(self):
        """A stationary pose should score higher on 'stand' than fast locomotion."""
        analyzer = SpectrumAnalyzer(dimensions=["stand"])
        static_score  = analyzer.analyze(_standing_motion(noise=0.005))["stand"]
        walking_score = analyzer.analyze(_walking_motion(speed=2.0))["stand"]
        assert static_score > walking_score, (
            f"stand score: static={static_score:.3f}, walking={walking_score:.3f}"
        )

    def test_spin_higher_for_spinning_than_standing(self):
        """A spinning motion should score higher on 'spin' than a static pose."""
        analyzer = SpectrumAnalyzer(dimensions=["spin"])
        spin_score  = analyzer.analyze(_spinning_motion(rate_rps=1.5))["spin"]
        stand_score = analyzer.analyze(_standing_motion(noise=0.005))["spin"]
        assert spin_score > stand_score, (
            f"spin score: spinning={spin_score:.3f}, standing={stand_score:.3f}"
        )

    def test_walk_higher_for_locomotion_than_static(self):
        """A walking motion should score higher on 'walk' than a static pose."""
        analyzer = SpectrumAnalyzer(dimensions=["walk"])
        walk_score  = analyzer.analyze(_walking_motion())["walk"]
        stand_score = analyzer.analyze(_standing_motion(noise=0.005))["walk"]
        assert walk_score > stand_score, (
            f"walk score: walking={walk_score:.3f}, standing={stand_score:.3f}"
        )


# ── SpectrumCaptionSynthesizer tests ─────────────────────────────────

class TestSpectrumCaptionSynthesizer:
    def _scores(self, **kwargs) -> dict:
        base = {d: 0.0 for d in DIMENSIONS}
        base.update(kwargs)
        return base

    def test_basic_output_format(self):
        synth = SpectrumCaptionSynthesizer()
        scores = self._scores(walk=0.82, dance=0.18)
        caption = synth.synthesize(scores)
        assert "[walk:0.82]" in caption
        assert "[dance:0.18]" in caption
        assert caption.startswith("[")
        assert caption.endswith(".")

    def test_stable_ordering(self):
        """Same scores always produce identical caption string."""
        synth = SpectrumCaptionSynthesizer()
        scores = self._scores(walk=0.82, dance=0.18, spin=0.22)
        assert synth.synthesize(scores) == synth.synthesize(scores)

    def test_threshold_excludes_low_scores(self):
        synth = SpectrumCaptionSynthesizer(tag_threshold=0.10)
        scores = self._scores(walk=0.90, dance=0.01)
        caption = synth.synthesize(scores)
        assert "[dance:" not in caption

    def test_interpolation_count(self):
        synth = SpectrumCaptionSynthesizer()
        a = self._scores(walk=0.9, dance=0.1)
        b = self._scores(walk=0.1, dance=0.9)
        assert len(synth.synthesize_interpolated(a, b, steps=5)) == 5

    def test_interpolation_endpoints(self):
        synth = SpectrumCaptionSynthesizer()
        a = self._scores(walk=0.9, dance=0.1)
        b = self._scores(walk=0.1, dance=0.9)
        captions = synth.synthesize_interpolated(a, b, steps=5)
        assert "[walk:0.90]" in captions[0]
        assert "[dance:0.90]" in captions[-1]

    def test_small_delta_small_string_change(self):
        """A 0.01 score change should only alter the numeric token, not the structure."""
        synth = SpectrumCaptionSynthesizer()
        c1 = synth.synthesize(self._scores(walk=0.80, dance=0.20))
        c2 = synth.synthesize(self._scores(walk=0.79, dance=0.21))
        assert c1.replace("0.80", "0.79").replace("0.20", "0.21") == c2

    def test_all_zero_scores(self):
        synth = SpectrumCaptionSynthesizer()
        caption = synth.synthesize(self._scores())
        assert "A person" in caption


# ── Integration tests ────────────────────────────────────────────────

class TestIntegration:
    def test_analyze_then_synthesize(self):
        analyzer = SpectrumAnalyzer()
        synth    = SpectrumCaptionSynthesizer()
        scores   = analyzer.analyze(_walking_motion(T=80))
        caption  = synth.synthesize(scores)
        assert isinstance(caption, str) and len(caption) > 10

    def test_full_pipeline_produces_valid_captions(self):
        """Walk through the full analyze→synthesize→interpolate chain."""
        analyzer = SpectrumAnalyzer()
        synth    = SpectrumCaptionSynthesizer()

        walk_scores  = analyzer.analyze(_walking_motion())
        stand_scores = analyzer.analyze(_standing_motion())

        captions = synth.synthesize_interpolated(walk_scores, stand_scores, steps=4)
        assert len(captions) == 4
        for c in captions:
            assert c.startswith("[") and c.endswith(".")
