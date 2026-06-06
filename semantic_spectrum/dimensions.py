"""
Kinematic feature extractors for each semantic dimension.

Input: raw joint coordinates, shape (T, 22, 3) — Y-up, units in meters, 20 fps.
No dependency on HumanML3D 263-dim preprocessing — operates directly on joint positions.
All extractors return a float in [0, 1].

HumanML3D / SMPL-H joint layout (22 joints):
  0  Pelvis          1  L_Hip           2  R_Hip
  3  Spine1          4  L_Knee          5  R_Knee
  6  Spine2          7  L_Ankle         8  R_Ankle
  9  Spine3         10  L_Toe          11  R_Toe
 12  Neck           13  L_Collar       14  R_Collar
 15  Head           16  L_Shoulder     17  R_Shoulder
 18  L_Elbow        19  R_Elbow        20  L_Wrist
 21  R_Wrist

Relationship to LMA / Hamscher et al. (arXiv:2511.20469):
  The 8 dimensions here are kinematic proxies intended to give the model a
  tractable conditioning signal.  They do not directly replicate LMA Effort/Shape
  qualities, but several map naturally:
    walk/run  ↔  Weight + Time (sustained, strong locomotion effort)
    dance     ↔  Space + Flow (indirect, free multi-limb coordination)
    spin      ↔  Shape: Rotation
    stand     ↔  Weight: Light / Time: Sustained (minimal effort)
  Hamscher's 168 LMA+FFT features could serve as a quantitative bridge between
  these kinematic scores and participant-facing LMA vocabulary in the user study.
"""

import numpy as np

FPS = 20.0  # HumanML3D capture frame rate

# ── Joint index constants ────────────────────────────────────────────
J_PELVIS = 0
J_L_HIP, J_R_HIP = 1, 2
J_SPINE1 = 3
J_L_KNEE, J_R_KNEE = 4, 5
J_SPINE2 = 6
J_L_ANKLE, J_R_ANKLE = 7, 8
J_SPINE3 = 9
J_L_TOE, J_R_TOE = 10, 11
J_NECK = 12
J_L_COLLAR, J_R_COLLAR = 13, 14
J_HEAD = 15
J_L_SHOULDER, J_R_SHOULDER = 16, 17
J_L_ELBOW, J_R_ELBOW = 18, 19
J_L_WRIST, J_R_WRIST = 20, 21

# Joint groups
FOOT_JOINTS       = [J_L_ANKLE, J_R_ANKLE, J_L_TOE, J_R_TOE]
LEFT_FOOT_JOINTS  = [J_L_ANKLE, J_L_TOE]
RIGHT_FOOT_JOINTS = [J_R_ANKLE, J_R_TOE]
ARM_JOINTS        = [J_L_SHOULDER, J_R_SHOULDER, J_L_ELBOW, J_R_ELBOW, J_L_WRIST, J_R_WRIST]
LEG_JOINTS        = [J_L_HIP, J_R_HIP, J_L_KNEE, J_R_KNEE, J_L_ANKLE, J_R_ANKLE, J_L_TOE, J_R_TOE]
UPPER_BODY_JOINTS = [J_SPINE1, J_SPINE2, J_SPINE3, J_NECK,
                     J_L_COLLAR, J_R_COLLAR, J_HEAD,
                     J_L_SHOULDER, J_R_SHOULDER,
                     J_L_ELBOW, J_R_ELBOW, J_L_WRIST, J_R_WRIST]


# ── Shared kinematic helpers ─────────────────────────────────────────

def _velocities(joints: np.ndarray, fps: float = FPS) -> np.ndarray:
    """Joint velocities. Returns (T-1, 22, 3) in m/s."""
    return np.diff(joints, axis=0) * fps


def _foot_contact(
    joints: np.ndarray,
    fps: float = FPS,
    height_margin: float = 0.08,
    vel_thresh: float = 1.5,
) -> np.ndarray:
    """
    Boolean contact mask of shape (T-1, 4): [L_ankle, R_ankle, L_toe, R_toe].

    Ground level is estimated as the minimum observed height of any foot joint
    across the whole sequence.  A joint is 'in contact' when it is near the
    floor AND moving slowly.
    """
    foot_pos = joints[:, FOOT_JOINTS, :]              # (T, 4, 3)
    ground_y = float(np.min(foot_pos[:, :, 1]))       # floor estimate
    foot_y   = foot_pos[:, :, 1]                      # (T, 4)
    foot_vel = np.linalg.norm(
        np.diff(foot_pos, axis=0) * fps, axis=-1
    )                                                  # (T-1, 4)
    low_height = foot_y[:-1] < ground_y + height_margin
    low_speed  = foot_vel < vel_thresh
    return low_height & low_speed                      # (T-1, 4)


def _root_xz_speed(joints: np.ndarray, fps: float = FPS) -> float:
    """Mean horizontal (XZ) speed of the pelvis in m/s."""
    disp = np.diff(joints[:, J_PELVIS, :], axis=0) * fps  # (T-1, 3)
    return float(np.mean(np.linalg.norm(disp[:, [0, 2]], axis=1)))


def _body_yaw_rate(joints: np.ndarray, fps: float = FPS) -> np.ndarray:
    """
    Angular velocity of body facing direction around the Y axis (rad/s).
    Derived from the hip-line direction projected onto the XZ plane.
    Returns shape (T-1,).
    """
    hip_vec = joints[:, J_R_HIP, :] - joints[:, J_L_HIP, :]  # (T, 3)
    hip_xz  = hip_vec[:, [0, 2]]                               # (T, 2)
    norms   = np.linalg.norm(hip_xz, axis=1, keepdims=True).clip(1e-6, None)
    hip_xz_n = hip_xz / norms                                  # (T, 2) unit vectors
    angle   = np.arctan2(hip_xz_n[:, 1], hip_xz_n[:, 0])     # (T,)
    return np.diff(np.unwrap(angle)) * fps                     # (T-1,) rad/s


# ── Semantic dimension extractors ───────────────────────────────────
# Each function takes (T, 22, 3) and returns float in [0, 1].

def extract_walk(joints: np.ndarray) -> float:
    """Steady horizontal locomotion with alternating foot contact and low pelvis bounce."""
    speed   = _root_xz_speed(joints)
    contact = _foot_contact(joints)                            # (T-1, 4)
    left    = contact[:, 0] | contact[:, 2]                   # any left-side contact
    right   = contact[:, 1] | contact[:, 3]
    alternation = float(np.mean(left ^ right))                 # one foot up, one down
    y_range = float(joints[:, J_PELVIS, 1].max() - joints[:, J_PELVIS, 1].min())

    score = (
        np.clip(speed / 2.5, 0, 1) * 0.35
        + np.clip(alternation / 0.6, 0, 1) * 0.40
        + np.clip(1.0 - y_range / 0.30, 0, 1) * 0.25
    )
    return float(np.clip(score, 0, 1))


def extract_dance(joints: np.ndarray) -> float:
    """Expressive upper-body articulation with rhythmic variation and trunk twist."""
    vels = _velocities(joints)                                 # (T-1, 22, 3)
    arm_speeds = np.linalg.norm(vels[:, ARM_JOINTS, :], axis=-1)  # (T-1, 6)
    arm_energy = float(np.mean(arm_speeds))
    # Variance in arm speed captures rhythmic change-of-speed bursts
    arm_var = float(np.var(arm_speeds))

    # Trunk twist: difference in azimuth between shoulder-line and hip-line
    hip_vec      = joints[:, J_R_HIP, :]      - joints[:, J_L_HIP, :]
    shoulder_vec = joints[:, J_R_SHOULDER, :] - joints[:, J_L_SHOULDER, :]
    hip_angle      = np.arctan2(hip_vec[:, 2],      hip_vec[:, 0])
    shoulder_angle = np.arctan2(shoulder_vec[:, 2], shoulder_vec[:, 0])
    twist_std = float(np.std(np.unwrap(shoulder_angle - hip_angle)))

    score = (
        np.clip(arm_energy / 2.0, 0, 1) * 0.40
        + np.clip(arm_var / 1.5, 0, 1) * 0.30
        + np.clip(twist_std / 0.35, 0, 1) * 0.30
    )
    return float(np.clip(score, 0, 1))


def extract_run(joints: np.ndarray) -> float:
    """High horizontal speed with periodic airborne phases."""
    speed   = _root_xz_speed(joints)
    contact = _foot_contact(joints)
    left    = contact[:, 0] | contact[:, 2]
    right   = contact[:, 1] | contact[:, 3]
    airborne = float(np.mean(~left & ~right))

    score = (
        np.clip(speed / 4.5, 0, 1) * 0.65
        + np.clip(airborne / 0.25, 0, 1) * 0.35
    )
    return float(np.clip(score, 0, 1))


def extract_jump(joints: np.ndarray) -> float:
    """Pelvis height peaks above resting level with simultaneous airborne phases."""
    contact = _foot_contact(joints)
    left    = contact[:, 0] | contact[:, 2]
    right   = contact[:, 1] | contact[:, 3]
    airborne_ratio = float(np.mean(~left & ~right))

    # Height above the 10th-percentile (resting) pelvis height
    pelvis_y    = joints[:, J_PELVIS, 1]
    rest_height = float(np.percentile(pelvis_y, 10))
    peak_lift   = float(np.percentile(pelvis_y - rest_height, 90))

    score = (
        np.clip(peak_lift / 0.30, 0, 1) * 0.55      # ~30 cm lift → full score
        + np.clip(airborne_ratio / 0.30, 0, 1) * 0.45
    )
    return float(np.clip(score, 0, 1))


def extract_spin(joints: np.ndarray) -> float:
    """Sustained rotation of the body around the vertical (Y) axis."""
    yaw_rate  = np.abs(_body_yaw_rate(joints))        # (T-1,) rad/s
    # 75th percentile rather than mean — captures sustained spinning
    sustained = float(np.percentile(yaw_rate, 75))
    # 2π rad/s ≈ one full revolution per second
    score = np.clip(sustained / (2 * np.pi), 0, 1)
    return float(score)


def extract_kick(joints: np.ndarray) -> float:
    """Peak leg velocity with asymmetric foot contact (one foot airborne)."""
    vels = _velocities(joints)                                 # (T-1, 22, 3)
    leg_speeds = np.linalg.norm(vels[:, LEG_JOINTS, :], axis=-1)  # (T-1, 8)
    peak_speed = float(np.percentile(leg_speeds, 95))

    contact = _foot_contact(joints)
    left    = contact[:, 0] | contact[:, 2]
    right   = contact[:, 1] | contact[:, 3]
    one_foot_up = float(np.mean(left ^ right))                 # exactly one foot grounded

    score = (
        np.clip(peak_speed / 6.0, 0, 1) * 0.55
        + np.clip(one_foot_up / 0.40, 0, 1) * 0.45
    )
    return float(np.clip(score, 0, 1))


def extract_wave(joints: np.ndarray) -> float:
    """High wrist velocity while the body root stays relatively still."""
    vels = _velocities(joints)                                 # (T-1, 22, 3)
    wrist_speeds = np.linalg.norm(
        vels[:, [J_L_WRIST, J_R_WRIST], :], axis=-1
    )                                                          # (T-1, 2)
    wrist_energy = float(np.mean(wrist_speeds))
    root_speed   = _root_xz_speed(joints)

    score = (
        np.clip(wrist_energy / 2.5, 0, 1) * 0.65
        + np.clip(1.0 - root_speed / 1.5, 0, 1) * 0.35
    )
    return float(np.clip(score, 0, 1))


def extract_stand(joints: np.ndarray) -> float:
    """Minimal whole-body velocity with both feet consistently grounded."""
    vels = _velocities(joints)                                 # (T-1, 22, 3)
    body_energy = float(np.mean(np.linalg.norm(vels, axis=-1)))

    contact = _foot_contact(joints)
    left    = contact[:, 0] | contact[:, 2]
    right   = contact[:, 1] | contact[:, 3]
    both_grounded = float(np.mean(left & right))

    score = (
        np.clip(1.0 - body_energy / 0.5, 0, 1) * 0.60
        + np.clip(both_grounded / 0.8, 0, 1) * 0.40
    )
    return float(np.clip(score, 0, 1))


# ── Registry ─────────────────────────────────────────────────────────
# name -> (extractor_fn, natural_language_descriptor)
DIMENSIONS = {
    "walk":  (extract_walk,  "walks forward"),
    "dance": (extract_dance, "dances with rhythmic movement"),
    "run":   (extract_run,   "runs"),
    "jump":  (extract_jump,  "jumps"),
    "spin":  (extract_spin,  "spins"),
    "kick":  (extract_kick,  "kicks"),
    "wave":  (extract_wave,  "waves"),
    "stand": (extract_stand, "stands"),
}
