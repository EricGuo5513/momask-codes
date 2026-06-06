"""
Semantic Motion Interpolation Experiment

Given a source motion and a target spectrum, generates a smooth sequence of
motions along the semantic manifold:

    [walk:0.90][dance:0.10] → [walk:0.70][dance:0.30] → ... → [walk:0.10][dance:0.90]

Usage:
    python scripts/semantic_interpolate.py \
        --source_motion example_data/000612.npy \
        --from_spec "walk:0.90,dance:0.10" \
        --to_spec   "walk:0.10,dance:0.90" \
        --steps 5 \
        --gpu_id 0 \
        --output_dir ./interpolation_results \
        --checkpoints_dir ./checkpoints

Outputs one .npy and optionally one .mp4 per step.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch


def parse_spec(spec_str: str) -> dict:
    """Parse 'walk:0.9,dance:0.1' -> {'walk': 0.9, 'dance': 0.1}"""
    result = {}
    for part in spec_str.split(","):
        k, v = part.strip().split(":")
        result[k.strip()] = float(v.strip())
    return result


def interpolate_specs(spec_a, spec_b, steps):
    keys = sorted(set(spec_a) | set(spec_b))
    specs = []
    for i in range(steps):
        t = i / max(steps - 1, 1)
        blended = {k: (1 - t) * spec_a.get(k, 0.0) + t * spec_b.get(k, 0.0) for k in keys}
        specs.append(blended)
    return specs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source_motion",   required=True)
    p.add_argument("--from_spec",       required=True, help="e.g. 'walk:0.9,dance:0.1'")
    p.add_argument("--to_spec",         required=True, help="e.g. 'walk:0.1,dance:0.9'")
    p.add_argument("--steps",           type=int, default=5)
    p.add_argument("--output_dir",      default="./interpolation_results")
    p.add_argument("--checkpoints_dir", default="./checkpoints")
    p.add_argument("--dataset",         default="t2m")
    p.add_argument("--gpu_id",          type=int, default=0)
    p.add_argument("--repeat_times",    type=int, default=1)
    p.add_argument("--visualize",       action="store_true")
    return p.parse_args()


def load_momask(checkpoints_dir, dataset, device):
    """Load pre-trained MoMask models (VQ + MaskTransformer + ResTransformer)."""
    from utils.get_opt import get_opt
    from models.vq.model import RVQVAE
    from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer

    vq_opt = get_opt(str(Path(checkpoints_dir) / dataset / "vq" / "opt.txt"), device)
    vq_model = RVQVAE(
        vq_opt,
        vq_opt.dim_pose,
        vq_opt.nb_code,
        vq_opt.code_dim,
        vq_opt.output_emb_width,
        vq_opt.down_t,
        vq_opt.stride_t,
        vq_opt.width,
        vq_opt.depth,
        vq_opt.dilation_growth_rate,
        vq_opt.vq_act,
        vq_opt.vq_norm,
    )
    ckpt = torch.load(
        str(Path(checkpoints_dir) / dataset / "vq" / "net_last.pth"),
        map_location=device,
    )
    vq_model.load_state_dict(ckpt["net"])
    vq_model.eval().to(device)

    t2m_opt = get_opt(str(Path(checkpoints_dir) / dataset / "t2m_transformer" / "opt.txt"), device)
    t2m_model = MaskTransformer(
        code_dim=t2m_opt.code_dim,
        cond_mode="text",
        latent_dim=t2m_opt.latent_dim,
        ff_size=t2m_opt.ff_size,
        num_layers=t2m_opt.num_layers,
        num_heads=t2m_opt.num_heads,
        dropout=t2m_opt.dropout,
        clip_dim=512,
        cond_drop_prob=t2m_opt.cond_drop_prob,
        clip_version=t2m_opt.clip_version,
        opt=t2m_opt,
    )
    ckpt = torch.load(
        str(Path(checkpoints_dir) / dataset / "t2m_transformer" / "net_best_fid.tar"),
        map_location=device,
    )
    t2m_model.load_state_dict(ckpt["t2m_transformer"], strict=False)
    t2m_model.eval().to(device)

    res_opt = get_opt(str(Path(checkpoints_dir) / dataset / "res_transformer" / "opt.txt"), device)
    res_model = ResidualTransformer(
        code_dim=res_opt.code_dim,
        cond_mode="text",
        latent_dim=res_opt.latent_dim,
        ff_size=res_opt.ff_size,
        num_layers=res_opt.num_layers,
        num_heads=res_opt.num_heads,
        dropout=res_opt.dropout,
        clip_dim=512,
        cond_drop_prob=res_opt.cond_drop_prob,
        clip_version=res_opt.clip_version,
        opt=res_opt,
    )
    ckpt = torch.load(
        str(Path(checkpoints_dir) / dataset / "res_transformer" / "net_best_fid.tar"),
        map_location=device,
    )
    res_model.load_state_dict(ckpt["res_transformer"], strict=False)
    res_model.eval().to(device)

    return vq_model, t2m_model, res_model, t2m_opt


def generate_from_caption(caption, m_length, vq_model, t2m_model, res_model, opt, device, repeat=1):
    """Generate motion tokens from caption, then decode."""
    from models.mask_transformer.tools import logits_scheduler

    with torch.no_grad():
        captions_batch = [caption] * repeat
        lengths_batch = torch.tensor([m_length] * repeat, device=device)

        # MaskTransformer generation
        mids = t2m_model.generate(captions_batch, lengths_batch // 4, opt.time_steps, opt.cond_scale)

        # ResidualTransformer refinement
        mids = res_model.generate(mids, captions_batch, lengths_batch // 4, 1, 1.0)

        # Decode from VQ codes to motion
        pred_motions = vq_model.forward_decoder(mids)  # (B, T, 263)

    return pred_motions.cpu().numpy()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    from semantic_spectrum.synthesizer import SpectrumCaptionSynthesizer
    synthesizer = SpectrumCaptionSynthesizer()

    spec_a = parse_spec(args.from_spec)
    spec_b = parse_spec(args.to_spec)
    specs = interpolate_specs(spec_a, spec_b, args.steps)
    captions = [synthesizer.synthesize(s) for s in specs]

    print("\nInterpolation trajectory:")
    for i, (s, c) in enumerate(zip(specs, captions)):
        print(f"  Step {i}: {c}")

    print("\nLoading MoMask models ...")
    try:
        vq_model, t2m_model, res_model, opt = load_momask(args.checkpoints_dir, args.dataset, device)
    except Exception as e:
        print(f"\n[WARNING] Could not load MoMask checkpoints: {e}")
        print("Captions have been printed above. Re-run after downloading checkpoints.")
        return

    source = np.load(args.source_motion)
    m_length = (len(source) // 4) * 4  # must be divisible by 4

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, caption in enumerate(captions):
        print(f"\nGenerating step {i}: {caption}")
        motions = generate_from_caption(
            caption, m_length,
            vq_model, t2m_model, res_model, opt, device,
            repeat=args.repeat_times,
        )
        for r in range(args.repeat_times):
            out_path = out_dir / f"step_{i:02d}_repeat_{r:02d}.npy"
            np.save(str(out_path), motions[r, :m_length])
            print(f"  Saved {out_path}")

        if args.visualize:
            try:
                from utils.plot_script import plot_3d_motion
                from utils.motion_process import recover_from_ric
                import utils.paramUtil as paramUtil

                motion_joints = recover_from_ric(
                    torch.tensor(motions[0, :m_length]).float(), 22
                ).numpy()
                save_path = str(out_dir / f"step_{i:02d}.mp4")
                plot_3d_motion(
                    save_path,
                    paramUtil.t2m_kinematic_chain,
                    motion_joints,
                    title=caption,
                    fps=20,
                )
                print(f"  Video -> {save_path}")
            except Exception as e:
                print(f"  Visualization failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
