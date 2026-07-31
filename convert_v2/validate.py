"""
validate.py
============

Standalone validator for the converter (humanml3d2ros4hri.py). Loads
the sample HumanML3D clip, converts it, and checks the result three ways:
numeric sanity (continuity, advisory-limit coverage, sign checks, gait
antiphase correlation), a full forward-kinematics reconstruction of the
source joint positions from the emitted angles (the ground-truth
end-to-end check of every axis/sign convention in the converter), and two
matplotlib renders (stick-figure comparison, top-down trajectory).

Run: python3 convert_v2/validate.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import humanml3d2ros4hri as c2

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(HERE, "..", "convert_from_pos", "sample0_repeat0_len164.npy")
FPS = 20.0
ELBOW_KNEE_ANKLE_HEAD_TARGET_M = 0.06

# ---------------------------------------------------------------------------
# skeleton connectivity, for the stick-figure render only
# ---------------------------------------------------------------------------

EDGES = [
    ("pelvis", "left_hip"), ("pelvis", "right_hip"), ("pelvis", "spine1"),
    ("spine1", "spine2"), ("spine2", "spine3"),
    ("spine3", "neck"), ("neck", "head"),
    ("spine3", "left_collar"), ("spine3", "right_collar"),
    ("left_collar", "left_shoulder"), ("right_collar", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("left_ankle", "left_foot"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"), ("right_ankle", "right_foot"),
]

TORSO_RIGID_JOINTS = [
    "spine1", "spine2", "spine3", "neck",
    "left_collar", "right_collar", "left_shoulder", "right_shoulder",
]

FK_TARGET_GROUP = [
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "head",
]


class FKModel:
    """Rigid-skeleton forward kinematics for the 24-DOF semantic skeleton.
    Bone lengths and rigid offsets are measured once from frame 0 of the
    source data and held fixed (standard rigid-retarget assumption)."""

    def __init__(self, j0: np.ndarray, a0: dict, yaw0: float):
        h3d = c2.H3D
        root_r0 = c2.root_frame_matrix(yaw0)
        pelvis0 = j0[h3d["pelvis"]]

        torso_local_r0 = (
            c2.rot_axis(c2.TORSO_AXIS1, a0["r_waist"])
            @ c2.rot_axis(c2.TORSO_AXIS2, a0["y_waist"])
            @ c2.rot_axis(c2.TORSO_AXIS3, a0["waist"])
        )
        torso_r0 = root_r0 @ torso_local_r0

        self.torso_offset = {
            name: torso_r0.T @ (j0[h3d[name]] - pelvis0) for name in TORSO_RIGID_JOINTS
        }
        self.hip_offset = {
            "l": root_r0.T @ (j0[h3d["left_hip"]] - pelvis0),
            "r": root_r0.T @ (j0[h3d["right_hip"]] - pelvis0),
        }
        self.head_len = float(np.linalg.norm(j0[h3d["head"]] - j0[h3d["neck"]]))

        def arm_lengths(side):
            sh = j0[h3d[f"{side}_shoulder"]]
            el = j0[h3d[f"{side}_elbow"]]
            wr = j0[h3d[f"{side}_wrist"]]
            return float(np.linalg.norm(el - sh)), float(np.linalg.norm(wr - el))

        def leg_lengths(side):
            hip = j0[h3d[f"{side}_hip"]]
            kn = j0[h3d[f"{side}_knee"]]
            ank = j0[h3d[f"{side}_ankle"]]
            foot = j0[h3d[f"{side}_foot"]]
            return (
                float(np.linalg.norm(kn - hip)),
                float(np.linalg.norm(ank - kn)),
                float(np.linalg.norm(foot - ank)),
            )

        self.upper_arm_len = {"l": arm_lengths("left")[0], "r": arm_lengths("right")[0]}
        self.forearm_len = {"l": arm_lengths("left")[1], "r": arm_lengths("right")[1]}
        thigh_l, shank_l, foot_l = leg_lengths("left")
        thigh_r, shank_r, foot_r = leg_lengths("right")
        self.thigh_len = {"l": thigh_l, "r": thigh_r}
        self.shank_len = {"l": shank_l, "r": shank_r}
        self.foot_len = {"l": foot_l, "r": foot_r}

    def solve(self, angles: dict, root_xy_yaw: tuple, pelvis_z: float) -> dict:
        root_x, root_y, yaw = root_xy_yaw
        pelvis_pos = np.array([root_x, root_y, pelvis_z])
        root_r = c2.root_frame_matrix(yaw)
        torso_local_r = (
            c2.rot_axis(c2.TORSO_AXIS1, angles["r_waist"])
            @ c2.rot_axis(c2.TORSO_AXIS2, angles["y_waist"])
            @ c2.rot_axis(c2.TORSO_AXIS3, angles["waist"])
        )
        torso_r = root_r @ torso_local_r

        pos = {"pelvis": pelvis_pos}
        for name in TORSO_RIGID_JOINTS:
            pos[name] = pelvis_pos + torso_r @ self.torso_offset[name]
        pos["left_hip"] = pelvis_pos + root_r @ self.hip_offset["l"]
        pos["right_hip"] = pelvis_pos + root_r @ self.hip_offset["r"]

        head_frame = (
            torso_r
            @ c2.rot_axis(c2.SPEC_HEAD.axis1, 0.0)
            @ c2.rot_axis(c2.SPEC_HEAD.axis2, angles["y_head"])
            @ c2.rot_axis(c2.SPEC_HEAD.axis3, angles["p_head"])
        )
        pos["head"] = pos["neck"] + self.head_len * (head_frame @ c2.UP)

        for side, tag in (("left", "l"), ("right", "r")):
            shoulder_frame = torso_r @ c2.limb_frame(
                c2.SPEC_ARM,
                (angles[f"{tag}_y_shoulder"], angles[f"{tag}_p_shoulder"], angles[f"{tag}_r_shoulder"]),
            )
            elbow_pos = pos[f"{side}_shoulder"] + self.upper_arm_len[tag] * (shoulder_frame @ c2.DOWN)
            pos[f"{side}_elbow"] = elbow_pos
            forearm_frame = shoulder_frame @ c2.rot_axis(
                c2.SPEC_ARM.hinge_rest_axis, angles[f"{tag}_elbow"]
            )
            pos[f"{side}_wrist"] = elbow_pos + self.forearm_len[tag] * (forearm_frame @ c2.DOWN)

        for side, tag, spec in (("left", "l", c2.SPEC_L_HIP), ("right", "r", c2.SPEC_R_HIP)):
            hip_frame = root_r @ c2.limb_frame(
                spec, (angles[f"{tag}_y_hip"], angles[f"{tag}_p_hip"], angles[f"{tag}_r_hip"])
            )
            knee_pos = pos[f"{side}_hip"] + self.thigh_len[tag] * (hip_frame @ c2.DOWN)
            pos[f"{side}_knee"] = knee_pos
            shank_frame = hip_frame @ c2.rot_axis(spec.hinge_rest_axis, angles[f"{tag}_knee"])
            ankle_pos = knee_pos + self.shank_len[tag] * (shank_frame @ c2.DOWN)
            pos[f"{side}_ankle"] = ankle_pos
            heading_pred = c2.ANKLE_REF_HEADING + angles[f"{tag}_ankle"]
            foot_dir_local = np.array([np.cos(heading_pred), 0.0, np.sin(heading_pred)])
            foot_dir_world = shank_frame @ foot_dir_local
            pos[f"{side}_foot"] = ankle_pos + self.foot_len[tag] * foot_dir_world

        return pos


def load_sample() -> np.ndarray:
    if not os.path.exists(SAMPLE_PATH):
        raise FileNotFoundError(f"sample file not found: {SAMPLE_PATH}")
    return np.load(SAMPLE_PATH)


def angles_matrix(frames: list) -> np.ndarray:
    return np.array([[f["angles"][name] for name in c2.ROS_JOINT_ORDER] for f in frames])


def print_table(header, rows, widths):
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt.format(*row))


def continuity_report(mat: np.ndarray) -> list:
    # wrap the diff first: a value near +pi followed by one near -pi is a
    # representation wraparound, not a real jump, and a raw diff would
    # misreport it as a ~2*pi discontinuity.
    raw_diff = np.diff(mat, axis=0)
    wrapped = (raw_diff + np.pi) % (2 * np.pi) - np.pi
    deltas = np.abs(wrapped)
    max_delta = deltas.max(axis=0)
    rows = []
    for i, name in enumerate(c2.ROS_JOINT_ORDER):
        flag = "FLAG" if max_delta[i] > 0.5 else ""
        rows.append((name, f"{max_delta[i]:.4f}", flag))
    return rows


def limits_report(mat: np.ndarray) -> list:
    rows = []
    for i, name in enumerate(c2.ROS_JOINT_ORDER):
        lo, hi = c2.LIMITS[name]
        col = mat[:, i]
        frac = float(np.mean((col < lo) | (col > hi)))
        rows.append((name, f"{frac * 100:.1f}%"))
    return rows


def sign_checks(mat: np.ndarray) -> None:
    idx = {name: i for i, name in enumerate(c2.ROS_JOINT_ORDER)}
    for name in ("l_elbow", "r_elbow"):
        bad = int(np.sum(mat[:, idx[name]] < -1e-9))
        print(f"  {name} < 0 in {bad}/{mat.shape[0]} frames (expect 0)")
    for name in ("l_knee", "r_knee"):
        bad = int(np.sum(mat[:, idx[name]] > 1e-9))
        print(f"  {name} > 0 in {bad}/{mat.shape[0]} frames (expect 0)")


def walking_segment(root_xy_yaw: list) -> tuple:
    xy = np.array([(r[0], r[1]) for r in root_xy_yaw])
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    speed = np.concatenate([[speed[0]], speed])
    thresh = 0.5 * float(np.median(speed))
    moving = speed > thresh
    best_start, best_len, cur_start, cur_len = 0, 0, 0, 0
    for i, m in enumerate(moving):
        if m:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
        else:
            cur_len = 0
    if best_len < 10:
        return 0, len(moving)
    return best_start, best_start + best_len


def antiphase_correlation(mat: np.ndarray, start: int, end: int) -> None:
    idx = {name: i for i, name in enumerate(c2.ROS_JOINT_ORDER)}
    pairs = [("l_p_shoulder", "l_p_hip"), ("r_p_shoulder", "r_p_hip")]
    for a, b in pairs:
        seg_a = mat[start:end, idx[a]]
        seg_b = mat[start:end, idx[b]]
        if np.std(seg_a) < 1e-9 or np.std(seg_b) < 1e-9:
            print(f"  corr({a}, {b}) over [{start},{end}): degenerate (no variance)")
            continue
        corr = float(np.corrcoef(seg_a, seg_b)[0, 1])
        print(f"  corr({a}, {b}) over [{start},{end}): {corr:+.3f} (negative = antiphase)")


def fk_reconstruction(data: np.ndarray, frames: list) -> dict:
    j = c2.h3d_to_ros_axes(data)
    model = FKModel(j[0], frames[0]["angles"], frames[0]["root_xy_yaw"][2])

    per_joint_errors = {name: [] for name in c2.H3D_JOINT_NAMES}
    fk_positions = []
    for t, f in enumerate(frames):
        pelvis_z = float(j[t][c2.H3D["pelvis"]][2])
        pred = model.solve(f["angles"], f["root_xy_yaw"], pelvis_z)
        fk_positions.append(pred)
        for name in c2.H3D_JOINT_NAMES:
            err = float(np.linalg.norm(pred[name] - j[t][c2.H3D[name]]))
            per_joint_errors[name].append(err)

    mean_errors = {name: float(np.mean(v)) for name, v in per_joint_errors.items()}
    return {"model": model, "fk_positions": fk_positions, "mean_errors": mean_errors}


def render_stick_figures(data: np.ndarray, fk_positions: list, frame_idx: list, out_path: str) -> None:
    j = c2.h3d_to_ros_axes(data)
    fig = plt.figure(figsize=(4 * len(frame_idx), 8))
    for col, t in enumerate(frame_idx):
        for row, (title, getter) in enumerate(
            [("source", lambda name, t=t: j[t][c2.H3D[name]]), ("FK recon", lambda name, t=t: fk_positions[t][name])]
        ):
            ax = fig.add_subplot(2, len(frame_idx), row * len(frame_idx) + col + 1, projection="3d")
            for a, b in EDGES:
                pa, pb = getter(a), getter(b)
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], color="tab:blue" if row == 0 else "tab:orange")
            ax.set_title(f"t={t} ({title})", fontsize=9)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(0, 2)
            ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def render_trajectory(frames: list, out_path: str) -> None:
    xy = np.array([f["root_xy_yaw"][:2] for f in frames])
    yaw = np.array([f["root_xy_yaw"][2] for f in frames])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xy[:, 0], xy[:, 1], color="tab:blue", lw=1.5, label="root path")
    step = max(1, len(frames) // 20)
    for i in range(0, len(frames), step):
        dx, dy = 0.1 * np.cos(yaw[i]), 0.1 * np.sin(yaw[i])
        ax.arrow(xy[i, 0], xy[i, 1], dx, dy, head_width=0.03, color="tab:red")
    ax.set_xlabel("x fwd (m)")
    ax.set_ylabel("y left (m)")
    ax.set_title("root trajectory + heading")
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    data = load_sample()
    print(f"loaded {data.shape} from {SAMPLE_PATH}")

    frames = c2.convert_sequence(data, fps=FPS, clamp_output=False)
    out_npy = os.path.join(HERE, "converted_v2.npy")
    np.save(out_npy, np.array(frames, dtype=object), allow_pickle=True)
    print(f"saved {out_npy}")

    mat = angles_matrix(frames)

    print("\n=== continuity (max |delta| per frame, rad) ===")
    print_table(("joint", "max_delta", "flag"), continuity_report(mat), (16, 10, 5))

    print("\n=== fraction of frames outside advisory limits ===")
    print_table(("joint", "frac_outside"), limits_report(mat), (16, 12))

    print("\n=== sign checks ===")
    sign_checks(mat)

    print("\n=== gait antiphase correlation ===")
    start, end = walking_segment([f["root_xy_yaw"] for f in frames])
    print(f"  walking segment: frames [{start}, {end})")
    antiphase_correlation(mat, start, end)

    print("\n=== FK reconstruction error (mean position error per joint, meters) ===")
    fk = fk_reconstruction(data, frames)
    rows = [(name, f"{err * 100:.2f} cm") for name, err in fk["mean_errors"].items()]
    print_table(("joint", "mean_error"), rows, (16, 12))
    group_mean = float(np.mean([fk["mean_errors"][n] for n in FK_TARGET_GROUP]))
    print(f"\n  target group (elbows/wrists/knees/ankles/head) mean: {group_mean * 100:.2f} cm "
          f"(target < {ELBOW_KNEE_ANKLE_HEAD_TARGET_M * 100:.0f} cm)")
    for name in FK_TARGET_GROUP:
        e = fk["mean_errors"][name]
        status = "OK" if e < ELBOW_KNEE_ANKLE_HEAD_TARGET_M else "OVER TARGET"
        print(f"    {name:<16} {e * 100:6.2f} cm  {status}")

    frame_idx = [0, 40, 80, 120, min(160, data.shape[0] - 1)]
    render_stick_figures(data, fk["fk_positions"], frame_idx, os.path.join(HERE, "validation_frames.png"))
    print(f"\nsaved {os.path.join(HERE, 'validation_frames.png')}")
    render_trajectory(frames, os.path.join(HERE, "validation_traj.png"))
    print(f"saved {os.path.join(HERE, 'validation_traj.png')}")

    reduced = c2.project_20dof(frames)
    print(f"\nproject_20dof: {len(reduced)} frames, {len(reduced[0]['angles'])} angle keys "
          f"(expect {len(c2.V1_JOINT_ORDER)})")


if __name__ == "__main__":
    main()
