"""Rough video-to-HumanML3D bridge for MoMask thesis demos.

This script performs a deliberately rough conversion from a user-uploaded
video to a HumanML3D-format .npy file suitable as the --source_motion input to
MoMask's edit_t2m.py. It is v0 scaffolding for a master's thesis demo, not a
faithful video-to-motion converter. It uses MediaPipe Pose 2D keypoints with
estimated z and approximates SMPL joint rotations via simple vector-geometry
heuristics. It does not do proper SMPL fitting. Proper SMPL fitting would
require WHAM, 4D-Humans, or HybrIK and is the next research milestone.
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

try:
    import mediapipe as mp
except ImportError as exc:
    raise ImportError(
        "MediaPipe is required. Install dependencies with: "
        "pip install mediapipe opencv-python numpy scipy"
    ) from exc


SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

DIRECT_MP_TO_SMPL = {
    23: 1,
    24: 2,
    25: 4,
    26: 5,
    27: 7,
    28: 8,
    31: 10,
    32: 11,
    11: 16,
    12: 17,
    13: 18,
    14: 19,
    15: 20,
    16: 21,
    0: 15,
}

SMPL_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19],
    dtype=np.int64,
)

CANONICAL_T_POSE = np.array(
    [
        [0.0, 0.0, 0.0],
        [-0.14, -0.10, 0.0],
        [0.14, -0.10, 0.0],
        [0.0, 0.18, 0.0],
        [0.0, -0.42, 0.0],
        [0.0, -0.42, 0.0],
        [0.0, 0.18, 0.0],
        [0.0, -0.42, 0.0],
        [0.0, -0.42, 0.0],
        [0.0, 0.18, 0.0],
        [0.0, -0.08, 0.12],
        [0.0, -0.08, 0.12],
        [0.0, 0.14, 0.0],
        [-0.10, 0.04, 0.0],
        [0.10, 0.04, 0.0],
        [0.0, 0.18, 0.0],
        [-0.18, 0.0, 0.0],
        [0.18, 0.0, 0.0],
        [-0.28, 0.0, 0.0],
        [0.28, 0.0, 0.0],
        [-0.25, 0.0, 0.0],
        [0.25, 0.0, 0.0],
    ],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Step 1: MediaPipe extraction
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roughly convert a video into a HumanML3D-style 263D .npy file."
    )
    parser.add_argument("--video", required=True, type=Path, help="Input video path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npy path. Defaults to video_bridge/<input_basename>.npy.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=20.0,
        help="Target FPS after nearest-neighbor resampling. Default: 20.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=196,
        help="Clip to at most this many output frames. Default: 196.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save a side-by-side MediaPipe overlay mp4 for extraction sanity checks.",
    )
    return parser.parse_args()


def read_video_and_extract_mediapipe(
    video_path: Path, visualize: bool
) -> tuple[np.ndarray, list[np.ndarray], float, tuple[int, int]]:
    if not video_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0.0 or math.isnan(fps):
        print("Warning: input FPS could not be read; assuming 30 FPS.")
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pose_cls = mp.solutions.pose.Pose
    all_landmarks: list[np.ndarray] = []
    frames: list[np.ndarray] = []

    with pose_cls(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if visualize:
                frames.append(frame_bgr.copy())

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)
            if result.pose_landmarks is None:
                all_landmarks.append(np.full((33, 4), np.nan, dtype=np.float32))
                continue

            frame_landmarks = np.array(
                [
                    [lm.x, lm.y, lm.z, lm.visibility]
                    for lm in result.pose_landmarks.landmark
                ],
                dtype=np.float32,
            )
            all_landmarks.append(frame_landmarks)

    cap.release()

    if not all_landmarks:
        raise RuntimeError(f"No frames were read from video: {video_path}")

    landmarks = np.stack(all_landmarks, axis=0)
    landmarks = interpolate_missing_pose_frames(landmarks)
    return landmarks, frames, fps, (width, height)


def interpolate_missing_pose_frames(landmarks: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(landmarks[:, 0, 0])
    missing_indices = np.where(~valid)[0]
    if missing_indices.size == 0:
        return landmarks

    runs = consecutive_runs(missing_indices)
    bad_runs = [run for run in runs if len(run) > 1]
    if bad_runs:
        first = bad_runs[0]
        raise RuntimeError(
            "MediaPipe failed to detect a pose for multiple consecutive frames "
            f"({first[0]}-{first[-1]}). Use a clearer full-body video or trim the clip."
        )

    fixed = landmarks.copy()
    for idx in missing_indices:
        prev_idx = idx - 1
        next_idx = idx + 1
        if prev_idx < 0 or next_idx >= len(landmarks) or not valid[prev_idx] or not valid[next_idx]:
            raise RuntimeError(
                "MediaPipe missed a pose at the start or end of the clip, so it "
                f"cannot be interpolated safely (frame {idx}). Trim the video."
            )
        fixed[idx] = 0.5 * (fixed[prev_idx] + fixed[next_idx])
        print(f"Warning: interpolated a missing MediaPipe pose at frame {idx}.")
    return fixed


def consecutive_runs(indices: Iterable[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for idx in indices:
        if not runs or idx != runs[-1][-1] + 1:
            runs.append([int(idx)])
        else:
            runs[-1].append(int(idx))
    return runs


def nearest_resample_indices(
    frame_count: int, original_fps: float, target_fps: float, max_frames: int
) -> np.ndarray:
    if target_fps <= 0.0:
        raise ValueError("--target-fps must be greater than 0.")
    if max_frames <= 0:
        raise ValueError("--max-frames must be greater than 0.")

    duration = frame_count / original_fps
    target_count = max(1, int(math.floor(duration * target_fps)))
    target_times = np.arange(target_count, dtype=np.float32) / target_fps
    indices = np.rint(target_times * original_fps).astype(np.int64)
    indices = np.clip(indices, 0, frame_count - 1)
    if len(indices) > max_frames:
        print(
            f"Warning: resampled sequence has {len(indices)} frames; "
            f"dropping frames after max_frames={max_frames}."
        )
        indices = indices[:max_frames]
    return indices


# ---------------------------------------------------------------------------
# Step 2: MediaPipe 33 -> SMPL 22 mapping
# ---------------------------------------------------------------------------


def mediapipe33_to_smpl22(mp_landmarks: np.ndarray) -> np.ndarray:
    smpl = np.zeros((len(mp_landmarks), 22, 4), dtype=np.float32)
    for mp_idx, smpl_idx in DIRECT_MP_TO_SMPL.items():
        smpl[:, smpl_idx] = mp_landmarks[:, mp_idx]

    left_hip = mp_landmarks[:, 23]
    right_hip = mp_landmarks[:, 24]
    left_shoulder = mp_landmarks[:, 11]
    right_shoulder = mp_landmarks[:, 12]
    pelvis = midpoint(left_hip, right_hip)
    shoulder_mid = midpoint(left_shoulder, right_shoulder)

    # These derived joints are crude. A real SMPL fitter would solve for them
    # via inverse kinematics over a parametric body model.
    smpl[:, 0] = pelvis
    smpl[:, 3] = lerp(pelvis, shoulder_mid, 1.0 / 3.0)
    smpl[:, 6] = lerp(pelvis, shoulder_mid, 2.0 / 3.0)
    smpl[:, 9] = shoulder_mid
    smpl[:, 12] = midpoint(left_shoulder, right_shoulder)
    smpl[:, 12, 1] -= 0.03
    smpl[:, 13] = lerp(smpl[:, 12], smpl[:, 16], 1.0 / 3.0)
    smpl[:, 14] = lerp(smpl[:, 12], smpl[:, 17], 1.0 / 3.0)
    return smpl


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * (a + b)


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


# ---------------------------------------------------------------------------
# Step 3: Coordinate normalization to HumanML3D conventions
# ---------------------------------------------------------------------------


def normalize_to_humanml3d(smpl_landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = smpl_landmarks[:, :, :3].copy()
    coords[:, :, 0] = coords[:, :, 0] - 0.5
    coords[:, :, 1] = 1.0 - coords[:, :, 1]
    coords[:, :, 2] = -coords[:, :, 2]

    left_hip = coords[:, 1]
    left_shoulder = coords[:, 16]
    proxy_dist = np.linalg.norm(left_shoulder - left_hip, axis=1)
    proxy_dist = proxy_dist[np.isfinite(proxy_dist) & (proxy_dist > 1e-6)]
    if proxy_dist.size == 0:
        print("Warning: scale proxy was invalid; using scale factor 1.0.")
        scale = 1.0
    else:
        scale = 0.5 / float(np.median(proxy_dist))

    coords *= scale
    pelvis = coords[:, 0].copy()
    centered = coords - pelvis[:, None, :]

    pelvis_delta = np.diff(pelvis[:, [0, 2]], axis=0, prepend=pelvis[:1, [0, 2]])
    hip_width = np.linalg.norm(coords[:, 1] - coords[:, 2], axis=1)
    drift_per_frame = float(np.nanmedian(np.abs(pelvis_delta[:, 1])))
    if drift_per_frame < 1e-4:
        drift_per_frame = float(np.nanmedian(hip_width) * 0.015)
    root_traj = np.zeros((len(coords), 3), dtype=np.float32)
    root_traj[:, 0] = np.cumsum(pelvis_delta[:, 0]).astype(np.float32)
    root_traj[:, 2] = (np.arange(len(coords), dtype=np.float32) * drift_per_frame).astype(
        np.float32
    )

    normalized = centered + root_traj[:, None, :]
    return normalized.astype(np.float32), root_traj.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 4: Build the HumanML3D 263-dim feature vector per frame
# ---------------------------------------------------------------------------


def build_humanml3d_features(joints: np.ndarray, root_traj: np.ndarray) -> np.ndarray:
    if joints.ndim != 3 or joints.shape[1:] != (22, 3):
        raise ValueError(f"Expected joints shape (T, 22, 3), got {joints.shape}.")

    root_y_velocity = np.zeros((len(joints), 1), dtype=np.float32)
    root_xz_velocity = np.diff(
        root_traj[:, [0, 2]], axis=0, prepend=root_traj[:1, [0, 2]]
    ).astype(np.float32)
    root_height = joints[:, 0:1, 1].astype(np.float32)
    root_features = np.concatenate(
        [root_y_velocity, root_xz_velocity, root_height], axis=1
    )

    local_positions = (joints[:, 1:] - joints[:, 0:1]).reshape(len(joints), 21 * 3)
    rotations_6d = estimate_local_rotations_6d(joints)
    local_velocities = np.diff(
        joints,
        axis=0,
        prepend=joints[:1],
    ).reshape(len(joints), 22 * 3)
    foot_contacts = estimate_foot_contacts(joints)

    features = np.concatenate(
        [
            root_features,
            local_positions.astype(np.float32),
            rotations_6d.astype(np.float32),
            local_velocities.astype(np.float32),
            foot_contacts.astype(np.float32),
        ],
        axis=1,
    )
    if features.shape[1] != 263:
        raise RuntimeError(f"Internal error: expected 263D features, got {features.shape}.")
    return features.astype(np.float32)


def estimate_local_rotations_6d(joints: np.ndarray) -> np.ndarray:
    canonical_dirs = canonical_bone_dirs()
    output = np.zeros((len(joints), 21, 6), dtype=np.float32)

    # This produces geometrically plausible but globally inconsistent rotations.
    # Joint hierarchy is not enforced; bone twist is undefined. A proper SMPL
    # fitter would solve all rotations jointly under a kinematic tree constraint.
    for frame_idx in range(len(joints)):
        for joint_idx in range(1, 22):
            parent_idx = int(SMPL_PARENTS[joint_idx])
            observed = joints[frame_idx, joint_idx] - joints[frame_idx, parent_idx]
            observed_norm = np.linalg.norm(observed)
            if observed_norm < 1e-8:
                rot_matrix = np.eye(3, dtype=np.float32)
            else:
                observed_dir = observed / observed_norm
                try:
                    rotation, _ = Rotation.align_vectors(
                        observed_dir[None, :], canonical_dirs[joint_idx][None, :]
                    )
                    rot_matrix = rotation.as_matrix().astype(np.float32)
                except ValueError:
                    rot_matrix = np.eye(3, dtype=np.float32)
            output[frame_idx, joint_idx - 1] = rot_matrix[:, :2].reshape(6)
    return output.reshape(len(joints), 21 * 6)


def canonical_bone_dirs() -> np.ndarray:
    dirs = np.zeros((22, 3), dtype=np.float32)
    for joint_idx in range(1, 22):
        parent_idx = int(SMPL_PARENTS[joint_idx])
        vec = CANONICAL_T_POSE[joint_idx] - CANONICAL_T_POSE[parent_idx]
        norm = np.linalg.norm(vec)
        dirs[joint_idx] = vec / norm if norm > 1e-8 else np.array([0.0, 1.0, 0.0])
    return dirs


def estimate_foot_contacts(joints: np.ndarray) -> np.ndarray:
    left_foot = joints[:, 10]
    right_foot = joints[:, 11]
    left_toe = joints[:, 7]
    right_toe = joints[:, 8]
    feet = np.stack([left_foot, right_foot, left_toe, right_toe], axis=1)

    heights = feet[:, :, 1]
    velocities = np.linalg.norm(
        np.diff(feet, axis=0, prepend=feet[:1]), axis=2
    )
    height_threshold = float(np.percentile(heights, 20))
    velocity_threshold = max(0.015, float(np.percentile(velocities, 25)) * 1.5)
    contacts = (heights <= height_threshold) & (velocities <= velocity_threshold)
    return contacts.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 5: Save and validate
# ---------------------------------------------------------------------------


def resolve_output_path(video_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return Path(__file__).resolve().parent / f"{video_path.stem}.npy"


def save_and_validate(features: np.ndarray, output_path: Path) -> Path:
    output_path_npz = output_path.with_suffix(".npz")
    output_path_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path_npz, motion=features.astype(np.float32))
    with np.load(output_path_npz) as saved_data:
        if "motion" not in saved_data:
            raise RuntimeError(f"Saved file is missing 'motion': {output_path_npz}")
        reloaded = saved_data["motion"]
    if reloaded.ndim != 2 or reloaded.shape[1] != 263:
        raise RuntimeError(f"Saved file has invalid shape: {reloaded.shape}")
    print(f"Frames processed: {reloaded.shape[0]}")
    print(f"Final shape: {reloaded.shape}")
    print(f"Output path: {output_path_npz}")
    print(
        "Note: this output is approximate. For research-grade results, replace "
        "with a proper SMPL fitter (WHAM, 4D-Humans, or HybrIK)."
    )
    return output_path_npz


# ---------------------------------------------------------------------------
# Step 6: Optional visualization
# ---------------------------------------------------------------------------


def save_overlay_video(
    frames: list[np.ndarray],
    landmarks: np.ndarray,
    sampled_indices: np.ndarray,
    output_path: Path,
    target_fps: float,
    size: tuple[int, int],
) -> None:
    if not frames:
        print("Warning: no frames were retained for visualization; skipping overlay.")
        return

    overlay_path = output_path.with_name(f"{output_path.stem}_overlay.mp4")
    width, height = size
    writer = cv2.VideoWriter(
        str(overlay_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        target_fps,
        (width * 2, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create overlay video: {overlay_path}")

    for source_idx in sampled_indices:
        frame = frames[int(source_idx)].copy()
        overlay = frame.copy()
        draw_skeleton(overlay, landmarks[int(source_idx)])
        side_by_side = np.concatenate([frame, overlay], axis=1)
        writer.write(side_by_side)
    writer.release()
    print(
        f"Overlay path: {overlay_path} "
        "(sanity-checks MediaPipe extraction only; it does not visualize MoMask input)."
    )


def draw_skeleton(frame: np.ndarray, landmarks: np.ndarray) -> None:
    height, width = frame.shape[:2]
    connections = mp.solutions.pose.POSE_CONNECTIONS
    points: list[tuple[int, int] | None] = []
    for x, y, _z, visibility in landmarks:
        if visibility < 0.2:
            points.append(None)
            continue
        px = int(np.clip(x * width, 0, width - 1))
        py = int(np.clip(y * height, 0, height - 1))
        points.append((px, py))

    for a, b in connections:
        if points[a] is not None and points[b] is not None:
            cv2.line(frame, points[a], points[b], (0, 255, 0), 2)
    for point in points:
        if point is not None:
            cv2.circle(frame, point, 3, (0, 128, 255), -1)


def main() -> None:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    output_path = resolve_output_path(video_path, args.output).expanduser().resolve()

    mp_landmarks_all, frames, original_fps, frame_size = read_video_and_extract_mediapipe(
        video_path, args.visualize
    )
    sampled_indices = nearest_resample_indices(
        len(mp_landmarks_all), original_fps, args.target_fps, args.max_frames
    )
    mp_landmarks = mp_landmarks_all[sampled_indices]
    smpl_landmarks = mediapipe33_to_smpl22(mp_landmarks)
    joints, root_traj = normalize_to_humanml3d(smpl_landmarks)
    features = build_humanml3d_features(joints, root_traj)
    saved_output_path = save_and_validate(features, output_path)

    if args.visualize:
        save_overlay_video(
            frames,
            mp_landmarks_all,
            sampled_indices,
            saved_output_path,
            args.target_fps,
            frame_size,
        )


if __name__ == "__main__":
    main()
