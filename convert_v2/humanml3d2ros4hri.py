"""
Converts HumanML3D (T, 22, 3) joint *positions* into the Arena human
skeleton joint contract (24 semantic joint angles, JOINTS.md in Arena's
task_generator). Solver core: bounded least-squares limb fit, hinge-plane
twist recovery, warm starts, sin(bend) confidence weighting.

Torso triple derivation: pelvis->spine3 alone cannot
separate r_waist from y_waist near upright (both only appear scaled by
sin(waist)). A second observation, the shoulder-across vector, has a rest
direction that sits on the root frame's local -Y axis (by construction of
the yaw itself) and depends on (r, y) but not w under R = Rx(r)Rz(y)Ry(w).
solve_ball fits both vectors jointly, recovering all three angles at once.
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

# ---------------------------------------------------------------------------
# 2. wire contract joint table
# ---------------------------------------------------------------------------
# limits are (lo, hi) in radians, advisory only (see clamp= option below).

LIMITS = {
    "r_waist": (-0.6, 0.6),
    "y_waist": (-0.8, 0.8),
    "waist": (-0.2, 1.0),
    "r_head": (-1.0, 1.0),
    "y_head": (-1.4, 1.4),
    "p_head": (-1.5, 1.5),
    "l_y_shoulder": (-3.1, 3.1),
    "l_p_shoulder": (-1.0, 3.3),
    "l_r_shoulder": (-1.6, 1.6),
    "l_elbow": (0.0, 2.5),
    "r_y_shoulder": (-3.1, 3.1),
    "r_p_shoulder": (-1.0, 3.3),
    "r_r_shoulder": (-1.6, 1.6),
    "r_elbow": (0.0, 2.5),
    "l_y_hip": (-0.1, 0.6),
    "l_p_hip": (-0.4, 3.3),
    "l_r_hip": (-0.4, 0.7),
    "l_knee": (-2.5, 0.0),
    "r_y_hip": (-0.1, 0.6),
    "r_p_hip": (-0.4, 3.3),
    "r_r_hip": (-0.4, 0.7),
    "r_knee": (-2.5, 0.0),
    "l_ankle": (-0.9, 0.6),
    "r_ankle": (-0.9, 0.6),
}

# publish-all order, matching GaitGenerator.JOINT_NAMES.
ROS_JOINT_ORDER = [
    "r_waist", "y_waist", "waist",
    "r_head", "y_head", "p_head",
    "l_y_shoulder", "l_p_shoulder", "l_r_shoulder", "l_elbow",
    "r_y_shoulder", "r_p_shoulder", "r_r_shoulder", "r_elbow",
    "l_y_hip", "l_p_hip", "l_r_hip", "l_knee",
    "r_y_hip", "r_p_hip", "r_r_hip", "r_knee",
    "l_ankle", "r_ankle",
]
assert set(ROS_JOINT_ORDER) == set(LIMITS)
assert len(ROS_JOINT_ORDER) == 24

# the 20 gait-driven joints, for project_20dof()
GAIT20_JOINT_ORDER = [
    "waist",
    "r_head", "y_head", "p_head",
    "l_y_shoulder", "l_p_shoulder", "l_r_shoulder", "l_elbow",
    "r_y_shoulder", "r_p_shoulder", "r_r_shoulder", "r_elbow",
    "l_y_hip", "l_p_hip", "l_r_hip", "l_knee",
    "r_y_hip", "r_p_hip", "r_r_hip", "r_knee",
]


def clamp(name: str, value: float) -> float:
    lo, hi = LIMITS[name]
    return float(np.clip(value, lo, hi))


# ---------------------------------------------------------------------------
# 3. vector / rotation helpers
# ---------------------------------------------------------------------------


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.zeros_like(v)
    return v / n


def rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix about an arbitrary unit axis."""
    axis = unit(np.asarray(axis, dtype=float))
    k = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


AX_X = np.array([1.0, 0.0, 0.0])
AX_Y = np.array([0.0, 1.0, 0.0])
AX_Z = np.array([0.0, 0.0, 1.0])
DOWN = np.array([0.0, 0.0, -1.0])  # rest direction of a hanging limb (arm/leg)
UP = np.array([0.0, 0.0, 1.0])  # rest direction of the neck->head and pelvis->chest bones
ACROSS_REST = np.array([0.0, -1.0, 0.0])  # rest direction of hip/shoulder across-lines


# ---------------------------------------------------------------------------
# 4. axis conversion: HumanML3D (X=left,Y=up,Z=forward) -> ROS4HRI (X=fwd,Y=left,Z=up)
# ---------------------------------------------------------------------------


def h3d_to_ros_axes(p: np.ndarray) -> np.ndarray:
    """p: (...,3) in HumanML3D axes -> (...,3) in ROS4HRI axes."""
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return np.stack([z, x, y], axis=-1)


# ---------------------------------------------------------------------------
# 5. root x/y/yaw (ROS-native: x fwd, y left, yaw CCW about +Z)
# ---------------------------------------------------------------------------


def compute_root_xy_yaw(frame_h3d: np.ndarray) -> tuple[float, float, float]:
    """frame_h3d: (22,3) raw HumanML3D-axis joint positions for one frame."""
    l_hip = frame_h3d[H3D["left_hip"]]
    r_hip = frame_h3d[H3D["right_hip"]]
    across = unit(r_hip - l_hip)
    forward = unit(np.cross(np.array([0.0, 1.0, 0.0]), across))
    yaw = float(np.arctan2(forward[0], forward[2]))
    root = frame_h3d[H3D["pelvis"]]
    root_x_ros = float(root[2])  # H3D forward (z) -> ROS forward (x)
    root_y_ros = float(root[0])  # H3D left (x) -> ROS left (y)
    return root_x_ros, root_y_ros, yaw


def root_frame_matrix(yaw: float) -> np.ndarray:
    """Orthonormal (forward,left,up) basis, as columns, for the given yaw,
    expressed in ROS4HRI world axes (X fwd, Y left, Z up)."""
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    left = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    return np.stack([fwd, left, up], axis=1)  # columns


def local(vec_world: np.ndarray, frame_r: np.ndarray) -> np.ndarray:
    return frame_r.T @ vec_world


# ---------------------------------------------------------------------------
# 6. generic 3-DoF "ball joint + following hinge" solver (rationale in the
#    convert_from_pos module docstring).
# ---------------------------------------------------------------------------


@dataclass
class LimbSpec:
    axis1: np.ndarray
    axis2: np.ndarray
    axis3: np.ndarray
    rest_dir: np.ndarray
    hinge_rest_axis: np.ndarray
    bounds1: tuple = (-np.pi, np.pi)
    bounds2: tuple = (-np.pi, np.pi)
    bounds3: tuple = (-np.pi, np.pi)


def limb_frame(spec: LimbSpec, angles: np.ndarray) -> np.ndarray:
    """The full parent-frame rotation matrix for a solved (a1,a2,a3) triple."""
    a1, a2, a3 = angles
    return rot_axis(spec.axis1, a1) @ rot_axis(spec.axis2, a2) @ rot_axis(spec.axis3, a3)


def solve_limb(
    spec: LimbSpec, v_bone: np.ndarray, v_next: np.ndarray | None, x0=(0.0, 0.0, 0.0)
) -> np.ndarray:
    """
    v_bone: observed unit vector proximal->mid, in parent frame coords.
    v_next: observed unit vector mid->distal, in parent frame coords
            (None if unavailable -> only the bone-direction residual is used).
    """
    have_hinge = v_next is not None and np.linalg.norm(v_next) > 1e-9
    weight = 0.0
    n_obs = np.zeros(3)
    if have_hinge:
        n_raw = np.cross(v_bone, v_next)
        n_norm = np.linalg.norm(n_raw)
        weight = float(np.clip(n_norm / 0.35, 0.0, 1.0))
        if n_norm > 1e-9:
            n_obs = n_raw / n_norm

    x0_arr = np.asarray(x0, dtype=float)
    reg = 0.2

    # axis1 is parallel to rest_dir for both the arm and hip specs (it is
    # the "twist about a near-vertical limb" axis), so when the observed
    # bone is itself close to rest_dir (limb hanging near-straight) axis1's
    # effect on pred_dir vanishes and it becomes as weakly observed as a
    # straight-limb roll -- same failure mode as the hinge-plane twist, so
    # it gets the same fix: scale its regularization up toward warm-start
    # as alignment with rest_dir grows, instead of leaving it free to drift
    # to a same-cost alternate branch (e.g. y_hip near 0 vs near pi).
    axis1_align = float(abs(np.dot(v_bone, unit(spec.axis1))))
    reg_a1 = reg + 1.5 * axis1_align

    def residual(angles):
        a1, a2, a3 = angles
        r = (
            rot_axis(spec.axis1, a1)
            @ rot_axis(spec.axis2, a2)
            @ rot_axis(spec.axis3, a3)
        )
        pred_dir = r @ spec.rest_dir
        res = list(pred_dir - v_bone)
        if weight > 0.0:
            pred_hinge = r @ spec.hinge_rest_axis
            n = n_obs if np.dot(n_obs, pred_hinge) >= 0 else -n_obs
            res += list(weight * (pred_hinge - n))
        else:
            res += [0.0, 0.0, 0.0]
        res.append(reg_a1 * (a1 - x0_arr[0]))
        res.append(reg * (a2 - x0_arr[1]))
        res.append(reg * (a3 - x0_arr[2]))
        return res

    lo = [spec.bounds1[0], spec.bounds2[0], spec.bounds3[0]]
    hi = [spec.bounds1[1], spec.bounds2[1], spec.bounds3[1]]
    x0_clipped = np.clip(np.asarray(x0, dtype=float), lo, hi)
    mid = [0.5 * (a + b) for a, b in zip(lo, hi)]
    best = None
    # trf can settle in a poor local minimum for the fully-coupled hip case
    # (axis3 not parallel to rest_dir, so all 3 angles interact). Retrying
    # from the bounds midpoint alongside the warm start and keeping whichever
    # achieves lower residual cost is cheap insurance against that.
    for start in (x0_clipped, np.clip(mid, lo, hi)):
        sol = least_squares(
            residual, x0=start, method="trf", bounds=(lo, hi), max_nfev=200
        )
        if best is None or sol.cost < best.cost:
            best = sol
    return best.x


def solve_ball(
    axis1: np.ndarray,
    axis2: np.ndarray,
    axis3: np.ndarray,
    rest1: np.ndarray,
    rest2: np.ndarray,
    obs1: np.ndarray,
    obs2: np.ndarray,
    bounds: tuple,
    x0=(0.0, 0.0, 0.0),
) -> np.ndarray:
    """3-DoF ball joint fit from two independent rest/observed vector pairs
    (rather than solve_limb's bone + hinge-plane-normal pair). Used for the
    torso triple, where both vectors are directly observed bone directions,
    not a hinge plane normal, so solve_limb's plane-normal machinery does
    not apply. Both vectors get equal weight (both are always well
    observed, no bend-confidence gating needed)."""
    x0_arr = np.asarray(x0, dtype=float)
    reg = 0.1
    lo = [b[0] for b in bounds]
    hi = [b[1] for b in bounds]

    def residual(angles):
        a1, a2, a3 = angles
        r = rot_axis(axis1, a1) @ rot_axis(axis2, a2) @ rot_axis(axis3, a3)
        res = list(r @ rest1 - obs1) + list(r @ rest2 - obs2)
        res += list(reg * (np.array([a1, a2, a3]) - x0_arr))
        return res

    x0_clipped = np.clip(x0_arr, lo, hi)
    sol = least_squares(
        residual, x0=x0_clipped, method="trf", bounds=(lo, hi), max_nfev=200
    )
    return sol.x


# ---------------------------------------------------------------------------
# 7. limb specs
# ---------------------------------------------------------------------------
# Solver bounds double as the numerical-stability guard the original
# solve_limb comment describes: axis1 of the arm spec, and axis1 of the hip
# spec, are parallel to rest_dir, so when the bone is close to that axis
# (near-vertical hanging arm, near-vertical thigh) axis1's effect on the
# bone-direction residual shrinks toward zero and the fit becomes weakly
# conditioned along that one axis, the same failure mode as the documented
# "roll on a straight limb" case. Arm/head use the advisory LIMITS as
# bounds: widening them to a full period reintroduces multi-cm FK drift.
# Hip y/r and torso bounds are padded wider than advisory instead, the
# advisory window saturates on dynamic motion (deep crouch, wide stride)
# and a pegged bound biases the whole chain.

# shoulder: SAME spec both sides (JOINTS.md section 1a, no mirroring). axis3
# is parallel to rest_dir (both (0,0,-1)) so it is a true twist: rotating
# about your own long axis does not move the bone tip, only the hinge-plane
# reference used to recover it via the elbow/wrist observation.

SPEC_ARM = LimbSpec(
    axis1=np.array([0.0, 0.0, 1.0]),  # y: azimuth about body-up
    axis2=np.array([0.0, -1.0, 0.0]),  # p: sagittal flexion
    axis3=np.array([0.0, 0.0, -1.0]),  # r: twist about the (rest) limb axis
    rest_dir=DOWN,
    hinge_rest_axis=np.array([0.0, -1.0, 0.0]),
    bounds1=LIMITS["l_y_shoulder"],
    bounds2=LIMITS["l_p_shoulder"],
    bounds3=LIMITS["l_r_shoulder"],
)

# hips: solver bounds for y_hip/r_hip are padded past the advisory range,
# the advisory window is tuned for mild walking and this fully-coupled fit
# (axis3 not parallel to rest_dir, unlike the arm) pegs at the bound for
# a majority of frames on more dynamic motion
# (crouching, wide stride), which biases the whole leg chain -- same
# saturation failure mode as the torso waist bound, same fix.
HIP_Y_BOUND = (-1.0, 1.0)
HIP_R_BOUND = (-1.0, 1.8)
SPEC_L_HIP = LimbSpec(
    axis1=np.array([0, 0, -1.0]),
    axis2=np.array([1.0, 0, 0]),
    axis3=np.array([0, -1.0, 0]),
    rest_dir=DOWN,
    hinge_rest_axis=np.array([0, -1.0, 0]),
    bounds1=HIP_Y_BOUND,
    bounds2=LIMITS["l_p_hip"],
    bounds3=HIP_R_BOUND,
)
SPEC_R_HIP = LimbSpec(
    axis1=np.array([0, 0, -1.0]),
    axis2=np.array([-1.0, 0, 0]),
    axis3=np.array([0, -1.0, 0]),
    rest_dir=DOWN,
    hinge_rest_axis=np.array([0, -1.0, 0]),
    bounds1=HIP_Y_BOUND,
    bounds2=LIMITS["r_p_hip"],
    bounds3=HIP_R_BOUND,
)

# head: roll (axis1) is unobservable from a single
# neck->head vector and is forced to 0.0 after solving, see convert_frame.
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

# torso: chain body -> r_waist(X) -> y_waist(Z) -> waist(Y) -> torso, per
# JOINTS.md torso triple (mirrors the head triple order: roll, yaw, pitch).
TORSO_AXIS1 = np.array([1.0, 0.0, 0.0])  # r_waist
TORSO_AXIS2 = np.array([0.0, 0.0, 1.0])  # y_waist
TORSO_AXIS3 = np.array([0.0, 1.0, 0.0])  # waist
TORSO_BOUNDS = ((-1.0, 1.0), (-1.2, 1.2), (-0.3, 2.0))


# ---------------------------------------------------------------------------
# 8. hinge (elbow/knee) angle -- pure angle-between, no ambiguity
# ---------------------------------------------------------------------------


def hinge_angle(v_bone: np.ndarray, v_next: np.ndarray, sign: float) -> float:
    """v_bone: proximal->mid. v_next: mid->distal.
    0 = straight, increasing magnitude = more flexed."""
    c = np.clip(np.dot(unit(v_bone), unit(v_next)), -1.0, 1.0)
    return float(sign * np.arccos(c))


# ---------------------------------------------------------------------------
# 9. ankle: signed sagittal hinge, referenced to a standing-pose calibration
# ---------------------------------------------------------------------------


def sagittal_heading(v_local: np.ndarray) -> float:
    """Heading of a vector within its frame's local X-Z (fwd-up) plane,
    i.e. the angle swept by rotation about the local -Y axis (the ankle/
    knee/elbow hinge family). Y component is dropped."""
    x, _y, z = v_local
    n = float(np.hypot(x, z))
    if n < 1e-9:
        return 0.0
    return float(np.arctan2(z / n, x / n))


def wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _synthetic_standing_pose() -> np.ndarray:
    """A rough standing, arms-down HumanML3D pose (X=left,Y=up,Z=forward),
    used only to calibrate the ankle rest heading (see ANKLE_REF_HEADING)."""
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


def _compute_ankle_ref_heading() -> float:
    """At the standing pose hip/knee angles are ~0 (bone colinear), so the
    shank frame equals root_R exactly, and no solver is needed here: the
    reference heading is just the standing pose's ankle->foot direction
    expressed in root-local X-Z coordinates. l/r are symmetric by
    construction of the synthetic pose, so a single left-side reading is
    used for both ankles."""
    pose = _synthetic_standing_pose()
    _rx, _ry, yaw = compute_root_xy_yaw(pose)
    root_r = root_frame_matrix(yaw)
    j = h3d_to_ros_axes(pose)
    foot_dir = unit(j[H3D["left_foot"]] - j[H3D["left_ankle"]])
    foot_local = local(foot_dir, root_r)
    return sagittal_heading(foot_local)


ANKLE_REF_HEADING = _compute_ankle_ref_heading()


def ankle_angle(shank_frame: np.ndarray, foot_dir_world: np.ndarray) -> float:
    """shank_frame: world-orientation matrix of the shank (hip_frame after
    the knee hinge is applied). foot_dir_world: observed ankle->foot unit
    vector in world (ROS) axes. Positive = dorsiflexion (toes up)."""
    foot_local = shank_frame.T @ foot_dir_world
    heading = sagittal_heading(foot_local)
    return wrap_pi(heading - ANKLE_REF_HEADING)


# ---------------------------------------------------------------------------
# 9b. shoulder (y, p) closed form + 1-DOF twist fit
# ---------------------------------------------------------------------------
# For SPEC_ARM, pred_dir(y, p) = Rz(y) @ R((0,-1,0), p) @ DOWN works out to
# (cos(y)sin(p), sin(y)sin(p), -cos(p)) -- plain spherical coordinates with
# pole at the shoulder's body-up axis. 2 unknowns exactly determine a point
# on S2, so (y, p) has an exact closed-form inverse. Running it through
# solve_limb's joint 3-parameter nonlinear fit alongside the hinge and
# regularization residuals was empirically landing in bad local minima
# (verified: residual ~0.2-0.4 on a target that has an exact zero-residual
# solution within bounds), which is what was driving the elbow/wrist FK
# error. The closed form has the standard spherical double cover: (y, p)
# and (y+pi, -p) map to the same direction. The branch closer to the warm
# start is kept, matching the contract's "extractors regularize via warm
# start" rule for the p=0 pole singularity. Only the twist r, which the
# direction alone cannot see, still needs the 1-parameter bounded fit.


def solve_arm_yp(v_bone: np.ndarray, x0_y: float, x0_p: float) -> tuple:
    v_bone = unit(v_bone)
    p_c = float(np.arccos(np.clip(-v_bone[2], -1.0, 1.0)))  # canonical, in [0, pi]
    sin_p = np.sin(p_c)
    if sin_p > 1e-6:
        y_c = float(np.arctan2(v_bone[1] / sin_p, v_bone[0] / sin_p))
    else:
        y_c = x0_y  # at the pole: azimuth is undefined, hold the warm start
    y_alt = wrap_pi(y_c + np.pi)
    p_alt = -p_c

    def dist(y, p):
        return abs(wrap_pi(y - x0_y)) + abs(p - x0_p)

    if dist(y_alt, p_alt) < dist(y_c, p_c):
        return y_alt, p_alt
    return y_c, p_c


def solve_arm_twist(
    y: float, p: float, v_next: np.ndarray | None, x0_r: float
) -> float:
    have_hinge = v_next is not None and np.linalg.norm(v_next) > 1e-9
    r_yp = rot_axis(SPEC_ARM.axis1, y) @ rot_axis(SPEC_ARM.axis2, p)
    weight = 0.0
    n_obs = np.zeros(3)
    if have_hinge:
        v_bone_dir = r_yp @ DOWN
        n_raw = np.cross(v_bone_dir, v_next)
        n_norm = np.linalg.norm(n_raw)
        weight = float(np.clip(n_norm / 0.35, 0.0, 1.0))
        if n_norm > 1e-9:
            n_obs = n_raw / n_norm
    reg = 0.2

    def residual(r_arr):
        (r,) = r_arr
        pred_hinge = r_yp @ rot_axis(SPEC_ARM.axis3, r) @ SPEC_ARM.hinge_rest_axis
        res = []
        if weight > 0.0:
            n = n_obs if np.dot(n_obs, pred_hinge) >= 0 else -n_obs
            res += list(weight * (pred_hinge - n))
        else:
            res += [0.0, 0.0, 0.0]
        res.append(reg * (r - x0_r))
        return res

    lo, hi = SPEC_ARM.bounds3
    x0_clipped = np.clip(x0_r, lo, hi)
    sol = least_squares(
        residual, x0=[x0_clipped], method="trf", bounds=([lo], [hi]), max_nfev=200
    )
    return float(sol.x[0])


# ---------------------------------------------------------------------------
# 10. per-frame conversion
# ---------------------------------------------------------------------------


class ConverterState:
    """Holds previous-frame solutions for warm starts."""

    def __init__(self):
        self.x0_torso = np.zeros(3)
        self.x0_head = np.zeros(3)
        self.x0_l_shoulder = np.zeros(3)
        self.x0_r_shoulder = np.zeros(3)
        self.x0_l_hip = np.zeros(3)
        self.x0_r_hip = np.zeros(3)


def convert_frame(
    frame_h3d: np.ndarray,
    state: ConverterState,
    t: float,
    animation_state: int = 0,
    clamp_output: bool = False,
) -> dict:
    """frame_h3d: (22,3) raw HumanML3D joint positions for a single frame."""
    frame_h3d = np.asarray(frame_h3d, dtype=float)

    root_x, root_y, yaw = compute_root_xy_yaw(frame_h3d)
    j = h3d_to_ros_axes(frame_h3d)
    root_r = root_frame_matrix(yaw)

    # ---- torso triple (r_waist, y_waist, waist) ----
    v_chest = local(unit(j[H3D["spine3"]] - j[H3D["pelvis"]]), root_r)
    v_across = local(unit(j[H3D["right_shoulder"]] - j[H3D["left_shoulder"]]), root_r)
    torso_sol = solve_ball(
        TORSO_AXIS1, TORSO_AXIS2, TORSO_AXIS3,
        UP, ACROSS_REST, v_chest, v_across,
        TORSO_BOUNDS, x0=state.x0_torso,
    )
    state.x0_torso = torso_sol
    r_waist_a, y_waist_a, waist_a = torso_sol
    torso_local_r = rot_axis(TORSO_AXIS1, r_waist_a) @ rot_axis(TORSO_AXIS2, y_waist_a) @ rot_axis(TORSO_AXIS3, waist_a)
    torso_r = root_r @ torso_local_r

    # ---- head (roll unobservable -> forced 0) ----
    v_head = local(unit(j[H3D["head"]] - j[H3D["neck"]]), torso_r)
    head_sol = solve_limb(SPEC_HEAD, v_head, None, x0=state.x0_head)
    state.x0_head = head_sol
    _r_head_raw, y_head_a, p_head_a = head_sol
    r_head_a = 0.0

    # ---- shoulders + elbows ----
    def solve_arm(shoulder_i, elbow_i, wrist_i, x0):
        v_bone = local(unit(j[elbow_i] - j[shoulder_i]), torso_r)
        v_next = local(unit(j[wrist_i] - j[elbow_i]), torso_r)
        y, p = solve_arm_yp(v_bone, x0[0], x0[1])
        r = solve_arm_twist(y, p, v_next, x0[2])
        sol = np.array([y, p, r])
        elbow = hinge_angle(j[elbow_i] - j[shoulder_i], j[wrist_i] - j[elbow_i], sign=+1.0)
        return sol, elbow

    l_sh_sol, l_elbow = solve_arm(
        H3D["left_shoulder"], H3D["left_elbow"], H3D["left_wrist"], state.x0_l_shoulder
    )
    state.x0_l_shoulder = l_sh_sol
    r_sh_sol, r_elbow = solve_arm(
        H3D["right_shoulder"], H3D["right_elbow"], H3D["right_wrist"], state.x0_r_shoulder
    )
    state.x0_r_shoulder = r_sh_sol

    # ---- hips + knees ----
    def solve_leg(spec, hip_i, knee_i, ankle_i, x0):
        v_bone = local(unit(j[knee_i] - j[hip_i]), root_r)
        v_next = local(unit(j[ankle_i] - j[knee_i]), root_r)
        sol = solve_limb(spec, v_bone, v_next, x0=x0)
        knee = hinge_angle(j[knee_i] - j[hip_i], j[ankle_i] - j[knee_i], sign=-1.0)
        return sol, knee

    l_hip_sol, l_knee = solve_leg(
        SPEC_L_HIP, H3D["left_hip"], H3D["left_knee"], H3D["left_ankle"], state.x0_l_hip
    )
    state.x0_l_hip = l_hip_sol
    r_hip_sol, r_knee = solve_leg(
        SPEC_R_HIP, H3D["right_hip"], H3D["right_knee"], H3D["right_ankle"], state.x0_r_hip
    )
    state.x0_r_hip = r_hip_sol

    # ---- ankles (need the shank world frame: hip rotation then knee hinge) ----
    def solve_ankle(spec, hip_sol, knee_val, ankle_i, foot_i):
        hip_frame = root_r @ limb_frame(spec, hip_sol)
        shank_frame = hip_frame @ rot_axis(spec.hinge_rest_axis, knee_val)
        foot_dir = unit(j[foot_i] - j[ankle_i])
        return ankle_angle(shank_frame, foot_dir)

    l_ankle_a = solve_ankle(SPEC_L_HIP, l_hip_sol, l_knee, H3D["left_ankle"], H3D["left_foot"])
    r_ankle_a = solve_ankle(SPEC_R_HIP, r_hip_sol, r_knee, H3D["right_ankle"], H3D["right_foot"])

    raw = {
        "r_waist": r_waist_a,
        "y_waist": y_waist_a,
        "waist": waist_a,
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
        "l_ankle": l_ankle_a,
        "r_ankle": r_ankle_a,
    }
    if clamp_output:
        angles = {name: clamp(name, raw[name]) for name in ROS_JOINT_ORDER}
    else:
        angles = {name: float(raw[name]) for name in ROS_JOINT_ORDER}

    return {
        "angles": angles,
        "root_xy_yaw": (root_x, root_y, yaw),
        "animation_state": int(animation_state),
        "t": float(t),
    }


# ---------------------------------------------------------------------------
# 11. full-sequence conversion + 20-DOF projection
# ---------------------------------------------------------------------------


def convert_sequence(
    joints_h3d_seq: np.ndarray,
    fps: float = 20.0,
    animation_state: int = 0,
    clamp_output: bool = False,
) -> list[dict]:
    """
    joints_h3d_seq: (T, 22, 3) HumanML3D joint positions.
    Returns: list (length T) of clip per-frame dicts (format in README.md).
    """
    joints_h3d_seq = np.asarray(joints_h3d_seq, dtype=float)
    if joints_h3d_seq.ndim != 3 or joints_h3d_seq.shape[1:] != (22, 3):
        raise ValueError(f"expected (T,22,3), got {joints_h3d_seq.shape}")

    state = ConverterState()
    out = []
    for i, frame in enumerate(joints_h3d_seq):
        out.append(
            convert_frame(
                frame, state, t=i / fps, animation_state=animation_state,
                clamp_output=clamp_output,
            )
        )
    return out


def project_20dof(frames: list[dict]) -> list[dict]:
    """Reduce clip frames to the 20 gait-driven joints, for
    A/B testing against the shipped pipeline: drops r_waist/y_waist/ankles,
    zeros the y/r shoulder DOFs (keeping p, sagittal flexion).
    root_xy_yaw/animation_state/t are passed through unchanged."""
    out = []
    for f in frames:
        a = f["angles"]
        v1_angles = {name: a[name] for name in GAIT20_JOINT_ORDER}
        v1_angles["l_y_shoulder"] = 0.0
        v1_angles["l_r_shoulder"] = 0.0
        v1_angles["r_y_shoulder"] = 0.0
        v1_angles["r_r_shoulder"] = 0.0
        out.append(
            {
                "angles": v1_angles,
                "root_xy_yaw": f["root_xy_yaw"],
                "animation_state": f["animation_state"],
                "t": f["t"],
            }
        )
    return out


# ---------------------------------------------------------------------------
# 12. smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) > 1:
        source_anim = np.load(sys.argv[1])
        print(f"Loaded {source_anim.shape} from {sys.argv[1]}")
    else:
        base = _synthetic_standing_pose()
        source_anim = np.stack([base, base])
        print("Using the synthetic standing pose (2 frames).")

    result = convert_sequence(source_anim, fps=20.0)
    np.save("converted_v2_demo.npy", np.array(result, dtype=object), allow_pickle=True)
    print(json.dumps(result[0], indent=2))
