"""
Convert new_joints/ (T, 22, 3) raw joint positions to new_joint_vecs/ (T, 263)
HumanML3D feature vectors required by the VQ tokenizer trainer.

Usage:
    python scripts/preprocess_joints.py \
        --data_root ./dataset/HumanML3D \
        --example_id 000021
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.skeleton import Skeleton
from utils.motion_process import process_file, recover_from_ric
from utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain


# ── t2m skeleton constants ────────────────────────────────────────────
L_IDX1, L_IDX2 = 5, 8
FID_R = [8, 11]
FID_L = [7, 10]
FACE_JOINT_INDX = [2, 1, 17, 16]
JOINTS_NUM = 22
FEET_THRE = 0.002


def _setup_globals(data_root: Path, example_id: str):
    """Inject skeleton globals that process_file reads from module scope."""
    import utils.motion_process as mp

    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    kinematic_chain = t2m_kinematic_chain

    # Find example file in subdirectory structure
    subdir = example_id[:2]
    example_path = data_root / "new_joints" / subdir / f"{example_id}.npy"
    if not example_path.exists():
        # Fallback: search all subdirs
        matches = list((data_root / "new_joints").rglob(f"{example_id}.npy"))
        if not matches:
            raise FileNotFoundError(f"Example file {example_id}.npy not found in {data_root}/new_joints")
        example_path = matches[0]

    example_data = np.load(example_path)
    example_data = example_data.reshape(len(example_data), -1, 3)
    example_tensor = torch.from_numpy(example_data)

    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    tgt_offsets = tgt_skel.get_offsets_joints(example_tensor[0])

    # Inject into motion_process module scope (process_file reads these as globals)
    mp.l_idx1 = L_IDX1
    mp.l_idx2 = L_IDX2
    mp.fid_r = FID_R
    mp.fid_l = FID_L
    mp.face_joint_indx = FACE_JOINT_INDX
    mp.n_raw_offsets = n_raw_offsets
    mp.kinematic_chain = kinematic_chain
    mp.tgt_offsets = tgt_offsets


def preprocess(data_root: str, example_id: str = "000021") -> None:
    data_root = Path(data_root)
    joints_dir = data_root / "new_joints"
    vecs_dir = data_root / "new_joint_vecs"
    vecs_dir.mkdir(exist_ok=True)

    _setup_globals(data_root, example_id)

    # Collect all real .npy files (skip macOS ._* resource forks)
    all_files = sorted(
        p for p in joints_dir.rglob("*.npy")
        if not p.name.startswith("._")
    )

    print(f"Found {len(all_files)} motion files. Saving to {vecs_dir}")

    skipped = 0
    for src in tqdm(all_files, desc="Processing"):
        dst = vecs_dir / src.name
        if dst.exists():
            continue
        try:
            source_data = np.load(src)[:, :JOINTS_NUM]
            data, _, _, _ = process_file(source_data, FEET_THRE)
            np.save(dst, data)
        except Exception as e:
            tqdm.write(f"  SKIP {src.name}: {e}")
            skipped += 1

    print(f"Done. Skipped {skipped}/{len(all_files)} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./dataset/HumanML3D")
    parser.add_argument("--example_id", default="000021",
                        help="Reference skeleton file ID (used to compute bone offsets)")
    args = parser.parse_args()
    preprocess(args.data_root, args.example_id)
