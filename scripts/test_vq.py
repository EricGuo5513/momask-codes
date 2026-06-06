"""
Quick test of the trained RVQVAE tokenizer.

Loads the checkpoint, runs encode→decode on a handful of validation motions,
and reports reconstruction quality metrics.

Usage:
    python scripts/test_vq.py --gpu_id 0
    python scripts/test_vq.py --gpu_id -1   # CPU
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.vq.model import RVQVAE
from utils.get_opt import get_opt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--name", default="RVQVAE")
    parser.add_argument("--dataset_name", default="t2m")
    parser.add_argument("--checkpoints_dir", default="./checkpoints")
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Number of val motions to test")
    args = parser.parse_args()

    device = torch.device("cpu" if args.gpu_id == -1 else f"cuda:{args.gpu_id}")

    # ── Load model ────────────────────────────────────────────────────
    ckpt_root = Path(args.checkpoints_dir) / args.dataset_name / args.name
    opt_path  = ckpt_root / "opt.txt"
    ckpt_path = ckpt_root / "model" / "net_best_fid.tar"
    if not ckpt_path.exists():
        ckpt_path = ckpt_root / "model" / "latest.tar"

    print(f"Loading opt from:        {opt_path}")
    print(f"Loading checkpoint from: {ckpt_path}")

    vq_opt = get_opt(str(opt_path), device)
    vq = RVQVAE(vq_opt,
                263,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.output_emb_width,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)

    ckpt = torch.load(ckpt_path, map_location=device)
    key = "vq_model" if "vq_model" in ckpt else "net"
    vq.load_state_dict(ckpt[key])
    vq.eval().to(device)
    print(f"Model loaded ({sum(p.numel() for p in vq.parameters())/1e6:.1f}M params)\n")

    # ── Load data ─────────────────────────────────────────────────────
    data_root = Path("./dataset/HumanML3D")
    mean = np.load(data_root / "Mean.npy")
    std  = np.load(data_root / "Std.npy")

    val_ids = (data_root / "val.txt").read_text().strip().splitlines()[:args.n_samples]

    vecs_dir = data_root / "new_joint_vecs"

    errors, perplexities = [], []

    print(f"{'ID':<12} {'frames':>6} {'recon_err':>10} {'perplexity':>11}")
    print("-" * 45)

    for mid in val_ids:
        npy = vecs_dir / f"{mid}.npy"
        if not npy.exists():
            continue
        raw = np.load(npy)
        # Normalise + window to multiple of 4 (required by temporal downsampling)
        T = (len(raw) // 4) * 4
        if T < 4:
            continue
        x = torch.from_numpy((raw[:T] - mean) / std).float().unsqueeze(0).to(device)

        with torch.no_grad():
            recon, commit_loss, perplexity = vq(x)

        err = (recon - x).abs().mean().item()
        errors.append(err)
        perplexities.append(perplexity.item())
        print(f"{mid:<12} {T:>6} {err:>10.4f} {perplexity.item():>11.1f}")

    print("-" * 45)
    print(f"{'MEAN':<12} {'':>6} {np.mean(errors):>10.4f} {np.mean(perplexities):>11.1f}")
    print()
    print("Interpretation:")
    print(f"  Recon error  {np.mean(errors):.4f}  (good < 0.10, great < 0.06)")
    print(f"  Perplexity   {np.mean(perplexities):.1f}  (good > 100, great > 300 for 512 codes)")


if __name__ == "__main__":
    main()
