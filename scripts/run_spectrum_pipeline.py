"""
Full Semantic Spectrum Pipeline Runner

Steps:
  1. (Optional) Fit calibration on the motion directory.
  2. Analyse all motions -> spectrum_scores.json
  3. Augment text_dir with synthetic spectrum captions.

Example:
    python scripts/run_spectrum_pipeline.py \
        --motion_dir ./HumanML3D/new_joint_vecs \
        --text_dir   ./HumanML3D/texts \
        --out_dir    ./semantic_spectrum_data \
        --split_file ./HumanML3D/train.txt \
        --fit_calibration \
        --n_synthetic 3 \
        --synthetic_prob 0.5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_spectrum.analyzer import SpectrumAnalyzer
from semantic_spectrum.augment import DatasetAugmenter
from semantic_spectrum.synthesizer import SpectrumCaptionSynthesizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--motion_dir",  required=True)
    p.add_argument("--text_dir",    required=True)
    p.add_argument("--out_dir",     default="./semantic_spectrum_data")
    p.add_argument("--split_file",  default=None, help="Optional: restrict to motions in this split file.")
    p.add_argument("--fit_calibration", action="store_true",
                   help="Fit per-dimension calibration before analysing (recommended for first run).")
    p.add_argument("--calibration_path", default=None,
                   help="Load existing calibration JSON instead of re-fitting.")
    p.add_argument("--n_synthetic", type=int, default=3,
                   help="Synthetic caption files per motion.")
    p.add_argument("--overwrite",   action="store_true")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    calibration_path = out_dir / "calibration.json"
    spectrum_cache_path = out_dir / "spectrum_scores.json"

    # ------------------------------------------------------------------ #
    # Step 1 — Calibration
    # ------------------------------------------------------------------ #
    if args.calibration_path:
        print(f"[1/3] Loading calibration from {args.calibration_path}")
        analyzer = SpectrumAnalyzer.load_calibration(args.calibration_path)
    elif args.fit_calibration:
        print("[1/3] Fitting calibration ...")
        analyzer = SpectrumAnalyzer()
        analyzer.fit_calibration(
            motion_dir=args.motion_dir,
            save_path=calibration_path,
        )
        print(f"      Calibration -> {calibration_path}")
    else:
        print("[1/3] No calibration — using raw extractor outputs.")
        analyzer = SpectrumAnalyzer()

    # ------------------------------------------------------------------ #
    # Step 2 — Analyse motions
    # ------------------------------------------------------------------ #
    print("[2/3] Analysing motions ...")
    scores = analyzer.analyze_directory(
        motion_dir=args.motion_dir,
        output_json=spectrum_cache_path,
        verbose=True,
    )
    print(f"      Cache -> {spectrum_cache_path}")

    # ------------------------------------------------------------------ #
    # Step 3 — Augment text directory
    # ------------------------------------------------------------------ #
    print("[3/3] Augmenting text directory ...")
    synthesizer = SpectrumCaptionSynthesizer()
    aug = DatasetAugmenter(
        motion_dir=args.motion_dir,
        text_dir=args.text_dir,
        spectrum_cache=spectrum_cache_path,
        analyzer=analyzer,
        synthesizer=synthesizer,
        n_synthetic=args.n_synthetic,
        seed=args.seed,
    )
    n = aug.run(split_file=args.split_file, overwrite=args.overwrite)
    print(f"      Done. {n} motions augmented.")

    # ------------------------------------------------------------------ #
    # Quick summary
    # ------------------------------------------------------------------ #
    sample_ids = list(scores.keys())[:3]
    print("\n--- Sample spectrum scores ---")
    for sid in sample_ids:
        print(f"  {sid}: {json.dumps(scores[sid], indent=None)}")

    print("\n--- Example captions ---")
    for sid in sample_ids:
        caption = synthesizer.synthesize(scores[sid])
        print(f"  {caption}")


if __name__ == "__main__":
    main()
