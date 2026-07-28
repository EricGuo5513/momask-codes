"""
h3d_to_ros4hri.py
==================

Convert HumanML3D (22, 3) per-frame joint *positions* into the ROS4HRI
20-DoF semantic-angle representation.

--------------------------------------------------------------------
WHY THIS IS AN IK PROBLEM, NOT A LOOKUP
--------------------------------------------------------------------
HumanML3D gives joint positions only. ROS4HRI
wants joint angles. Some ROS4HRI joints are fully determined by a single
bone direction (2 DoF from a unit vector -> e.g. shoulder yaw+pitch,
waist lean). Others are NOT: the shoulder/hip "roll" DoF is a rotation
about the limb's own long axis, which a single bone vector cannot see at
all -- rotating a bone around its own axis doesn't move its endpoint.

To recover that missing DoF we use the next joint down the chain
(elbow uses the wrist, shoulder uses the elbow+wrist) as a second
reference point, exactly the way IK retargeting tools (Mixamo, VRM,
etc.) do it: the plane formed by (shoulder, elbow, wrist) tells you how
the forearm is "twisted" relative to the upper arm, which constrains the
otherwise-invisible roll angle.

Where even that isn't possible (head roll: HumanML3D's 22 joints have no
ear/eye/chin markers past the head joint itself), we cannot recover the
DoF and default it to 0, clearly flagged below. If your data has more
head landmarks, that's the one function to extend.

--------------------------------------------------------------------
COORDINATE CONVENTIONS
--------------------------------------------------------------------
HumanML3D (from raw_offsets in the prompt: spine offsets are +Y, hip
offsets are +-X, foot/head offsets are +Z) uses:
    X = left(+) / right(-)
    Y = up
    Z = forward

ROS4HRI wants:
    X = forward
    Y = left
    Z = up

so the axis remap is the cyclic permutation
    ROS.x = H3D.z
    ROS.y = H3D.x
    ROS.z = H3D.y
(a proper rotation, determinant +1, handedness preserved).

All the joint-local rotation axes below are taken verbatim from the
conversion table (they are defined THE WAY URDF <axis> tags are: each
joint's axis lives in its own preceding link's frame, so a 3-joint
chain like l_y_shoulder -> l_p_shoulder -> l_r_shoulder is a genuine
Euler-angle-style rotation chain, composed in that order).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy.optimize import least_squares

# ---------------------------------------------------------------------------
# 1. HumanML3D skeleton definition
# ---------------------------------------------------------------------------

H3D_JOINT_NAMES = [
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
H3D = {name: i for i, name in enumerate(H3D_JOINT_NAMES)}

KINEMATIC_CHAINS = [
    [0, 2, 5, 8, 11],  # right leg
    [0, 1, 4, 7, 10],  # left leg
    [0, 3, 6, 9, 12, 15],  # spine -> head
    [9, 14, 17, 19, 21],  # right arm
    [9, 13, 16, 18, 20],  # left arm
]

# ---------------------------------------------------------------------------
# 2. ROS4HRI joint table
# ---------------------------------------------------------------------------
# limits are (lo, hi) in radians, used to clamp final output.

LIMITS = {
    "waist": (-0.2, 1.0),
    "r_head": (-1.0, 1.0),
    "y_head": (-1.4, 1.4),
    "p_head": (-1.5, 1.5),
    "l_y_shoulder": (-1.1, 1.9),
    "l_p_shoulder": (-0.4, 3.3),
    "l_r_shoulder": (-1.7, 1.5),
    "l_elbow": (0.0, 2.5),
    "r_y_shoulder": (-1.1, 1.9),
    "r_p_shoulder": (-0.4, 3.3),
    "r_r_shoulder": (-1.7, 1.5),
    "r_elbow": (0.0, 2.5),
    "l_y_hip": (-0.1, 0.6),
    "l_p_hip": (-0.4, 3.3),
    "l_r_hip": (-0.4, 0.7),
    "l_knee": (-2.5, 0.0),
    "r_y_hip": (-0.1, 0.6),
    "r_p_hip": (-0.4, 3.3),
    "r_r_hip": (-0.4, 0.7),
    "r_knee": (-2.5, 0.0),
}

ROS_JOINT_ORDER = list(LIMITS.keys())  # the 20 output keys, table order


def clamp(name: str, value: float) -> float:
    lo, hi = LIMITS[name]
    return float(np.clip(value, lo, hi))


# ---------------------------------------------------------------------------
# 3. small vector / rotation helpers
# ---------------------------------------------------------------------------


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.zeros_like(v)
    return v / n


def rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix about an arbitrary unit axis."""
    axis = unit(np.asarray(axis, dtype=float))
    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


AX_X = np.array([1.0, 0.0, 0.0])
AX_Y = np.array([0.0, 1.0, 0.0])
AX_Z = np.array([0.0, 0.0, 1.0])
DOWN = np.array([0.0, 0.0, -1.0])  # rest direction of a hanging limb (arm/leg)
UP = np.array([0.0, 0.0, 1.0])  # rest direction of the neck->head bone


# ---------------------------------------------------------------------------
# 4. axis conversion: HumanML3D (X=left,Y=up,Z=forward) -> ROS4HRI (X=fwd,Y=left,Z=up)
# ---------------------------------------------------------------------------


def h3d_to_ros_axes(p: np.ndarray) -> np.ndarray:
    """p: (...,3) in HumanML3D axes -> (...,3) in ROS4HRI axes."""
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return np.stack([z, x, y], axis=-1)


# ---------------------------------------------------------------------------
# 5. root x/z/yaw  (computed in HumanML3D's OWN ground-plane convention,
#    i.e. this is the root trajectory, kept separate from the local pose
#    exactly the way HumanML3D's own representation splits root motion
#    from joint rotations)
# ---------------------------------------------------------------------------


def compute_root_xz_yaw(frame_h3d: np.ndarray) -> tuple[float, float, float]:
    """frame_h3d: (22,3) raw HumanML3D-axis joint positions for one frame."""
    l_hip = frame_h3d[H3D["left_hip"]]
    r_hip = frame_h3d[H3D["right_hip"]]
    across = r_hip - l_hip
    across = unit(across)
    forward = unit(np.cross(np.array([0.0, 1.0, 0.0]), across))
    yaw = float(np.arctan2(forward[0], forward[2]))
    root = frame_h3d[H3D["pelvis"]]
    return float(root[0]), float(root[2]), yaw


def root_frame_matrix(yaw: float) -> np.ndarray:
    """Orthonormal (forward,left,up) basis, as columns, for the given yaw,
    expressed in ROS4HRI world axes (X fwd, Y left, Z up)."""
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    left = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    return np.stack([fwd, left, up], axis=1)  # columns


# ---------------------------------------------------------------------------
# 6. generic 3-DoF "ball joint + following hinge" solver
# ---------------------------------------------------------------------------
#
# Models a chain   parent_frame -> Rot(axis1,a1) -> Rot(axis2,a2) -> Rot(axis3,a3) -> bone
# where `bone` points from the proximal joint (e.g. shoulder) to the mid
# joint (e.g. elbow) with rest direction `rest_dir` at (0,0,0), and the
# hinge joint at the mid joint (e.g. elbow) has rest axis `hinge_rest_axis`
# in that same frame. We fit (a1,a2,a3) to match:
#   (a) the observed bone direction (proximal->mid), and
#   (b) the observed hinge-plane normal (from proximal->mid->distal), which
#       is our stand-in observation for the hinge axis.
#
# This uniformly handles both the shoulder/elbow case (where axis3 is
# parallel to rest_dir, so a3="roll" is a true twist invisible to (a) alone
# and only recoverable via (b)) and the hip/knee case (where axis3 is NOT
# parallel to rest_dir, so all three angles are coupled) -- no per-joint
# hand-derived formula needed, one solver covers both.


@dataclass
class LimbSpec:
    axis1: np.ndarray
    axis2: np.ndarray
    axis3: np.ndarray
    rest_dir: np.ndarray
    hinge_rest_axis: np.ndarray
    # physical joint limits for (a1,a2,a3), in the SAME order as axis1/2/3.
    # Critical: without these, an under-constrained angle (e.g. roll on a
    # near-straight limb, where the residual gradient is almost flat) can
    # drift to an arbitrary, effectively-random multiple of 2*pi under an
    # unbounded solver -- clamping the final result then just clips that huge
    # number to a limit, which LOOKS like a plausible angle but isn't one.
    bounds1: tuple = (-np.pi, np.pi)
    bounds2: tuple = (-np.pi, np.pi)
    bounds3: tuple = (-np.pi, np.pi)


def solve_limb(
    spec: LimbSpec, v_bone: np.ndarray, v_next: np.ndarray | None, x0=(0.0, 0.0, 0.0)
) -> np.ndarray:
    """
    v_bone: observed unit vector proximal->mid, in parent frame coords.
    v_next: observed unit vector mid->distal, in parent frame coords
            (None if unavailable -> only the bone-direction residual is used,
            angle3 is then only weakly constrained and stays near x0).
    """
    # Confidence weight for the hinge-plane (twist) constraint: when the limb
    # is nearly straight (elbow/knee ~0), the shoulder/elbow/wrist plane is
    # ill-defined and its normal is dominated by numerical noise. We weight
    # by sin(bend_angle) = |cross(v_bone, v_next)|, which is exactly the
    # quantity that vanishes as the limb straightens, so the roll/twist
    # solution smoothly falls back to the warm-started previous value
    # instead of being dragged onto a noisy, poorly-conditioned plane.
    have_hinge = v_next is not None and np.linalg.norm(v_next) > 1e-9
    weight = 0.0
    n_obs = np.zeros(3)
    if have_hinge:
        n_raw = np.cross(v_bone, v_next)
        n_norm = np.linalg.norm(n_raw)
        weight = float(
            np.clip(n_norm / 0.35, 0.0, 1.0)
        )  # full confidence by ~20 deg of bend
        if n_norm > 1e-9:
            n_obs = n_raw / n_norm

    x0_arr = np.asarray(x0, dtype=float)
    REG = 0.2  # small relative to a fully-confident (weight=1) hinge residual,
    # but large enough to dominate and hold the angle near x0 when hinge
    # confidence is low (nearly-straight limb) instead of chasing plane noise

    def residual(angles):
        a1, a2, a3 = angles
        R = (
            rot_axis(spec.axis1, a1)
            @ rot_axis(spec.axis2, a2)
            @ rot_axis(spec.axis3, a3)
        )
        pred_dir = R @ spec.rest_dir
        res = list(pred_dir - v_bone)
        if weight > 0.0:
            pred_hinge = R @ spec.hinge_rest_axis
            n = n_obs if np.dot(n_obs, pred_hinge) >= 0 else -n_obs
            res += list(weight * (pred_hinge - n))
        else:
            res += [0.0, 0.0, 0.0]
        res += list(REG * (np.array([a1, a2, a3]) - x0_arr))
        return res

    lo = [spec.bounds1[0], spec.bounds2[0], spec.bounds3[0]]
    hi = [spec.bounds1[1], spec.bounds2[1], spec.bounds3[1]]
    x0_clipped = np.clip(np.asarray(x0, dtype=float), lo, hi)
    sol = least_squares(
        residual, x0=x0_clipped, method="trf", bounds=(lo, hi), max_nfev=200
    )
    return sol.x


# ---------------------------------------------------------------------------
# 7. Limb specs (axes copied verbatim from the conversion table)
# ---------------------------------------------------------------------------

SPEC_L_SHOULDER = LimbSpec(
    axis1=np.array([0, 0, -1.0]),
    axis2=np.array([1.0, 0, 0]),
    axis3=np.array([0, 0, 1.0]),
    rest_dir=DOWN,
    hinge_rest_axis=np.array([0, -1.0, 0]),
    bounds1=LIMITS["l_y_shoulder"],
    bounds2=LIMITS["l_p_shoulder"],
    bounds3=LIMITS["l_r_shoulder"],
)
SPEC_R_SHOULDER = LimbSpec(
    axis1=np.array([0, 0, 1.0]),
    axis2=np.array([-1.0, 0, 0]),
    axis3=np.array([0, 0, -1.0]),
    rest_dir=DOWN,
    hinge_rest_axis=np.array([0, -1.0, 0]),
    bounds1=LIMITS["r_y_shoulder"],
    bounds2=LIMITS["r_p_shoulder"],
    bounds3=LIMITS["r_r_shoulder"],
)
SPEC_L_HIP = LimbSpec(
    axis1=np.array([0, 0, -1.0]),
    axis2=np.array([1.0, 0, 0]),
    axis3=np.array([0, -1.0, 0]),
    rest_dir=DOWN,
    hinge_rest_axis=np.array([0, -1.0, 0]),
    bounds1=LIMITS["l_y_hip"],
    bounds2=LIMITS["l_p_hip"],
    bounds3=LIMITS["l_r_hip"],
)
SPEC_R_HIP = LimbSpec(
    axis1=np.array([0, 0, -1.0]),
    axis2=np.array([-1.0, 0, 0]),
    axis3=np.array([0, -1.0, 0]),
    rest_dir=DOWN,
    hinge_rest_axis=np.array([0, -1.0, 0]),
    bounds1=LIMITS["r_y_hip"],
    bounds2=LIMITS["r_p_hip"],
    bounds3=LIMITS["r_r_hip"],
)
# Head has no point past the head joint, so it only ever gets the bone-direction
# residual (2 DoF worth of information); the 3rd angle (roll, forced to 0.0
# after solving -- see convert_frame) simply stays wherever the bounded
# solver leaves it, which is harmless since we overwrite it anyway.
SPEC_HEAD = LimbSpec(
    axis1=np.array([1.0, 0, 0]),
    axis2=np.array([0, 0, 1.0]),
    axis3=np.array([0, -1.0, 0]),
    rest_dir=UP,
    hinge_rest_axis=np.array([0, 0, 0]),  # unused (no hinge)
    bounds1=LIMITS["r_head"],
    bounds2=LIMITS["y_head"],
    bounds3=LIMITS["p_head"],
)


# ---------------------------------------------------------------------------
# 8. hinge (elbow/knee) angle -- pure angle-between, no ambiguity
# ---------------------------------------------------------------------------


def hinge_angle(v_bone: np.ndarray, v_next: np.ndarray, sign: float) -> float:
    """v_bone: proximal->mid (e.g. shoulder->elbow). v_next: mid->distal
    (e.g. elbow->wrist). 0 = straight, increasing magnitude = more flexed."""
    c = np.clip(np.dot(unit(v_bone), unit(v_next)), -1.0, 1.0)
    return float(sign * np.arccos(c))


# ---------------------------------------------------------------------------
# 9. waist -- single DoF, closed form
# ---------------------------------------------------------------------------


def waist_angle(v_pelvis_to_chest_root_local: np.ndarray) -> float:
    """v_pelvis_to_chest_root_local: unit(chest-pelvis) expressed in the
    yaw-only root frame. Rest direction is straight up (0,0,1); rotation
    is about the root's local Y ("left") axis, axis (0,1,0) per the table."""
    lx, _ly, lz = v_pelvis_to_chest_root_local
    return float(np.arctan2(lx, lz))


# ---------------------------------------------------------------------------
# 10. torso (chest) frame -- parent frame for head + shoulders
# ---------------------------------------------------------------------------


def torso_frame_matrix(root_R: np.ndarray, waist_ang: float) -> np.ndarray:
    """root_R rotated by the waist lean about its own local Y (left) axis."""
    left_axis_world = root_R[:, 1]
    return root_R @ rot_axis(np.array([0.0, 1.0, 0.0]), waist_ang)
    # NOTE: rot_axis with axis (0,1,0) is applied in root_R's *local* frame
    # sense here because we right-multiply root_R; rot_axis itself is built
    # from the canonical (0,1,0) axis which, when right-multiplied like this,
    # is interpreted in root_R's local coordinates -- exactly what we want.


# ---------------------------------------------------------------------------
# 11. Per-frame conversion
# ---------------------------------------------------------------------------


class ConverterState:
    """Holds previous-frame solutions so the numeric solver gets a warm
    start (keeps sequences temporally smooth and converges faster)."""

    def __init__(self):
        self.x0_l_shoulder = np.zeros(3)
        self.x0_r_shoulder = np.zeros(3)
        self.x0_l_hip = np.zeros(3)
        self.x0_r_hip = np.zeros(3)
        self.x0_head = np.zeros(3)


def convert_frame(
    frame_h3d: np.ndarray, state: ConverterState, t: float, animation_state: int = 0
) -> dict:
    """frame_h3d: (22,3) raw HumanML3D joint positions for a single frame."""
    frame_h3d = np.asarray(frame_h3d, dtype=float)

    # --- root trajectory, computed in H3D's own ground convention ---
    root_x, root_z, yaw = compute_root_xz_yaw(frame_h3d)

    # --- everything else happens in ROS4HRI axes ---
    j = h3d_to_ros_axes(frame_h3d)  # (22,3)
    root_R = root_frame_matrix(yaw)

    def local(vec_world, frame_R):
        return frame_R.T @ vec_world

    # ---- waist ----
    v_chest = unit(j[H3D["spine3"]] - j[H3D["pelvis"]])
    w_local = local(v_chest, root_R)
    w_ang = waist_angle(w_local)
    w_ang_c = clamp("waist", w_ang)
    torso_R = torso_frame_matrix(root_R, w_ang_c)

    # ---- head (roll unobservable -> defaults near 0 via solver init) ----
    v_head = local(unit(j[H3D["head"]] - j[H3D["neck"]]), torso_R)
    sol = solve_limb(SPEC_HEAD, v_head, None, x0=state.x0_head)
    state.x0_head = sol
    r_head_a, y_head_a, p_head_a = sol
    r_head_a, y_head_a, p_head_a = 0.0, y_head_a, p_head_a  # roll unobservable: force 0

    # ---- shoulders + elbows ----
    def solve_arm(spec, shoulder_i, elbow_i, wrist_i, x0):
        v_bone = local(unit(j[elbow_i] - j[shoulder_i]), torso_R)
        v_next = local(unit(j[wrist_i] - j[elbow_i]), torso_R)
        sol = solve_limb(spec, v_bone, v_next, x0=x0)
        elbow = hinge_angle(
            j[elbow_i] - j[shoulder_i], j[wrist_i] - j[elbow_i], sign=+1.0
        )
        return sol, elbow

    l_sh_sol, l_elbow = solve_arm(
        SPEC_L_SHOULDER,
        H3D["left_shoulder"],
        H3D["left_elbow"],
        H3D["left_wrist"],
        state.x0_l_shoulder,
    )
    state.x0_l_shoulder = l_sh_sol
    r_sh_sol, r_elbow = solve_arm(
        SPEC_R_SHOULDER,
        H3D["right_shoulder"],
        H3D["right_elbow"],
        H3D["right_wrist"],
        state.x0_r_shoulder,
    )
    state.x0_r_shoulder = r_sh_sol

    # ---- hips + knees ----
    def solve_leg(spec, hip_i, knee_i, ankle_i, x0):
        v_bone = local(unit(j[knee_i] - j[hip_i]), root_R)
        v_next = local(unit(j[ankle_i] - j[knee_i]), root_R)
        sol = solve_limb(spec, v_bone, v_next, x0=x0)
        knee = hinge_angle(j[knee_i] - j[hip_i], j[ankle_i] - j[knee_i], sign=-1.0)
        return sol, knee

    l_hip_sol, l_knee = solve_leg(
        SPEC_L_HIP, H3D["left_hip"], H3D["left_knee"], H3D["left_ankle"], state.x0_l_hip
    )
    state.x0_l_hip = l_hip_sol
    r_hip_sol, r_knee = solve_leg(
        SPEC_R_HIP,
        H3D["right_hip"],
        H3D["right_knee"],
        H3D["right_ankle"],
        state.x0_r_hip,
    )
    state.x0_r_hip = r_hip_sol

    raw = {
        "waist": w_ang_c,
        "r_head": r_head_a,
        "y_head": y_head_a,
        "p_head": p_head_a,
        "l_y_shoulder": l_sh_sol[0],
        "l_p_shoulder": l_sh_sol[1],
        "l_r_shoulder": l_sh_sol[2],
        "l_elbow": l_elbow,
        "r_y_shoulder": r_sh_sol[0],
        "r_p_shoulder": r_sh_sol[1],
        "r_r_shoulder": r_sh_sol[2],
        "r_elbow": r_elbow,
        "l_y_hip": l_hip_sol[0],
        "l_p_hip": l_hip_sol[1],
        "l_r_hip": l_hip_sol[2],
        "l_knee": l_knee,
        "r_y_hip": r_hip_sol[0],
        "r_p_hip": r_hip_sol[1],
        "r_r_hip": r_hip_sol[2],
        "r_knee": r_knee,
    }
    angles = {name: clamp(name, raw[name]) for name in ROS_JOINT_ORDER}

    return {
        "angles": angles,
        "root_xz_yaw": (root_x, root_z, yaw),
        "animation_state": int(animation_state),
        "t": float(t),
    }


# ---------------------------------------------------------------------------
# 12. Full-sequence conversion
# ---------------------------------------------------------------------------


def convert_sequence(
    joints_h3d_seq: np.ndarray, fps: float = 20.0, animation_state: int = 0
) -> list[dict]:
    """
    joints_h3d_seq: (T, 22, 3) HumanML3D joint positions.
    Returns: list (length T) of per-frame dicts:
        {"angles": {...20 keys...}, "root_xz_yaw": (x,z,yaw),
         "animation_state": int, "t": float}
    """
    joints_h3d_seq = np.asarray(joints_h3d_seq, dtype=float)
    assert joints_h3d_seq.ndim == 3 and joints_h3d_seq.shape[1:] == (22, 3), (
        f"expected (T,22,3), got {joints_h3d_seq.shape}"
    )

    state = ConverterState()
    out = []
    for i, frame in enumerate(joints_h3d_seq):
        out.append(
            convert_frame(frame, state, t=i / fps, animation_state=animation_state)
        )
    return out


# ---------------------------------------------------------------------------
# 13. Demo / smoke test
# ---------------------------------------------------------------------------


def _synthetic_standing_pose() -> np.ndarray:
    """A rough standing, arms-down HumanML3D pose (X=left,Y=up,Z=forward)."""
    p = np.zeros((22, 3))
    p[H3D["pelvis"]] = (0, 1.00, 0)
    p[H3D["left_hip"]] = (0.10, 0.90, 0)
    p[H3D["right_hip"]] = (-0.10, 0.90, 0)
    p[H3D["spine1"]] = (0, 1.10, 0)
    p[H3D["left_knee"]] = (0.10, 0.50, 0)
    p[H3D["right_knee"]] = (-0.10, 0.50, 0)
    p[H3D["spine2"]] = (0, 1.30, 0)
    p[H3D["left_ankle"]] = (0.10, 0.10, 0)
    p[H3D["right_ankle"]] = (-0.10, 0.10, 0)
    p[H3D["spine3"]] = (0, 1.45, 0)
    p[H3D["left_foot"]] = (0.10, 0.00, 0.10)
    p[H3D["right_foot"]] = (-0.10, 0.00, 0.10)
    p[H3D["neck"]] = (0, 1.55, 0)
    p[H3D["left_collar"]] = (0.08, 1.50, 0)
    p[H3D["right_collar"]] = (-0.08, 1.50, 0)
    p[H3D["head"]] = (0, 1.70, 0)
    p[H3D["left_shoulder"]] = (0.18, 1.48, 0)
    p[H3D["right_shoulder"]] = (-0.18, 1.48, 0)
    p[H3D["left_elbow"]] = (0.20, 1.20, 0)
    p[H3D["right_elbow"]] = (-0.20, 1.20, 0)
    p[H3D["left_wrist"]] = (0.22, 0.95, 0)
    p[H3D["right_wrist"]] = (-0.22, 0.95, 0)
    return p


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) > 1:
        source_anim = np.load(sys.argv[1])
        print(f"Loaded {source_anim.shape} from {sys.argv[1]}")
    else:
        base = _synthetic_standing_pose()
        # a few frames: standing, then raising the (rigid, straight) left arm
        # out to the side via shoulder abduction only -- an anatomically
        # valid motion, unlike bending the elbow sideways.
        shoulder = base[H3D["left_shoulder"]]
        upper_len = np.linalg.norm(base[H3D["left_elbow"]] - shoulder)
        fore_len = np.linalg.norm(base[H3D["left_wrist"]] - base[H3D["left_elbow"]])
        frames = []
        for t in range(5):
            f = base.copy()
            lift = t / 4.0
            ang = lift * (np.pi / 2)  # 0 -> 90 deg abduction
            # direction swings from straight down (0,-1,0) toward sideways (1,0,0)
            # in the (X,Y) = (left/right, up) plane, i.e. H3D axes (x,y)
            d = np.array([np.sin(ang), -np.cos(ang), 0.0])
            f[H3D["left_elbow"]] = shoulder + upper_len * d
            f[H3D["left_wrist"]] = f[H3D["left_elbow"]] + fore_len * d
            frames.append(f)
        source_anim = np.stack(frames)
        print(
            "Using built-in synthetic demo sequence (standing -> raising straight left arm sideways)."
        )

    result = convert_sequence(source_anim, fps=20.0)
    np.save("converted_anim.npy", result)
    print(json.dumps(result, indent=2))
