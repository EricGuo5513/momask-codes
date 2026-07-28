"""
Conversion from HumanML3D 263-dimension representation to ROS4HRI semantic
angle format.

Design notes / assumptions (see accompanying review):
  * HumanML3D's rotation block (indices [130:256]) stores a 6D continuous
    rotation for each of the 21 *non-root* joints, and each joint's 6D
    rotation is already local, i.e. relative to its own direct parent in
    the standard 22-joint SMPL/HumanML3D kinematic tree. The root (pelvis,
    joint 0) itself has no stored orientation.
  * ROS4HRI joints don't always coincide with direct HumanML3D parents
    (e.g. head's ROS4HRI ancestor is "torso" == HumanML3D joint 9, but
    head's real HumanML3D parent is "neck" == joint 12). Where there is
    exactly one unmapped joint in between (neck under head, collar under
    shoulders), we compose local rotations along the physical chain to
    get the rotation relative to the ROS4HRI ancestor.
  * "waist" (forward lean) has no literal source in the 263-dim vector,
    since root orientation isn't stored. We approximate it using spine1
    (HumanML3D joint 3), whose local rotation is relative to the root.
    This is a deliberate approximation -- swap ROOT_PROXY_JOINT / plug in
    a properly reconstructed root orientation if you have one.
  * "torso" is a fixed joint per the ROS4HRI tree (no DOF), so it is
    excluded from the output, matching the original DOF table.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from scipy.spatial.transform import Rotation


class HumanML3D2ROS4HRIConverter:
    ROT_START = 130  # start index of the 6D joint-rotation block in the 263-dim vector

    # ROS4HRI joint group name -> HumanML3D joint index
    HML3D_ROS4HRI_MAP: ClassVar[dict[str, int]] = {
        "waist": 0,
        "head": 15,
        "l_shoulder": 16,
        "l_elbow": 18,
        "r_shoulder": 17,
        "r_elbow": 19,
        "l_hip": 1,
        "l_knee": 4,
        "r_hip": 2,
        "r_knee": 5,
        "torso": 9,
    }

    # Standard 22-joint HumanML3D/SMPL kinematic tree: parent id per joint id.
    # (pelvis, l_hip, r_hip, spine1, l_knee, r_knee, spine2, l_ankle, r_ankle,
    #  spine3, l_foot, r_foot, neck, l_collar, r_collar, head, l_shoulder,
    #  r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist)
    HML3D_PARENTS: ClassVar[list[int]] = [
        -1,
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        9,
        9,
        12,
        13,
        14,
        16,
        17,
        18,
        19,
    ]

    # For each mapped HumanML3D joint id, the nearest ROS4HRI-mapped ancestor
    # joint id. Derived from HML3D_PARENTS + which joints are ROS4HRI-mapped.
    ROS4HRI_ANCESTOR: ClassVar[dict[int, int]] = {
        15: 9,  # head          -> torso  (through neck=12)
        16: 9,  # l_shoulder    -> torso  (through l_collar=13)
        17: 9,  # r_shoulder    -> torso  (through r_collar=14)
        18: 16,  # l_elbow       -> l_shoulder (direct parent)
        19: 17,  # r_elbow       -> r_shoulder (direct parent)
        1: 0,  # l_hip         -> waist/root (direct parent)
        2: 0,  # r_hip         -> waist/root (direct parent)
        4: 1,  # l_knee        -> l_hip (direct parent)  [fixed: was root]
        5: 2,  # r_knee        -> r_hip (direct parent)  [fixed: was root]
        9: 0,  # torso         -> waist/root
    }

    ROOT_PROXY_JOINT = 0  # pelvis: used as a stand-in source for "waist"

    # Single-DOF joint groups: (dof_name, unit axis (sign encodes direction), (lo, hi))
    SINGLE_DOF_SPEC: ClassVar[dict[str, tuple]] = {
        "waist": ("waist", (0.0, 1.0, 0.0), (-0.2, 1.0)),
        "l_elbow": ("l_elbow", (0.0, -1.0, 0.0), (0.0, 2.5)),
        "r_elbow": ("r_elbow", (0.0, -1.0, 0.0), (0.0, 2.5)),
        "l_knee": ("l_knee", (0.0, -1.0, 0.0), (-2.5, 0.0)),
        "r_knee": ("r_knee", (0.0, -1.0, 0.0), (-2.5, 0.0)),
    }

    # Multi-DOF joint groups: physical rotation order (proximal -> distal per
    # the ROS4HRI kinematic tree), pre-derived scipy Euler sequence letters
    # and per-axis signs from the table's axis vectors, plus limits.
    #   dofs: list of (name, sign, (lo, hi)) in `seq` order
    MULTI_DOF_SPEC: ClassVar[dict[str, dict]] = {
        "head": {
            "seq": "XZY",  # r_head(X) -> y_head(Z) -> p_head(Y, flipped)
            "dofs": [
                ("r_head", 1, (-1.0, 1.0)),
                ("y_head", 1, (-1.4, 1.4)),
                ("p_head", -1, (-1.5, 1.5)),
            ],
        },
        "l_shoulder": {
            "seq": "ZXZ",  # l_y_shoulder(Z,flip) -> l_p_shoulder(X) -> l_r_shoulder(Z)
            "dofs": [
                ("l_y_shoulder", -1, (-1.1, 1.9)),
                ("l_p_shoulder", 1, (-0.4, 3.3)),
                ("l_r_shoulder", 1, (-1.7, 1.5)),
            ],
        },
        "r_shoulder": {
            "seq": "ZXZ",  # r_y_shoulder(Z) -> r_p_shoulder(X,flip) -> r_r_shoulder(Z,flip)
            "dofs": [
                ("r_y_shoulder", 1, (-1.1, 1.9)),
                ("r_p_shoulder", -1, (-0.4, 3.3)),
                ("r_r_shoulder", -1, (-1.7, 1.5)),
            ],
        },
        "l_hip": {
            "seq": "ZXY",  # l_y_hip(Z,flip) -> l_p_hip(X) -> l_r_hip(Y,flip)
            "dofs": [
                ("l_y_hip", -1, (-0.1, 0.6)),
                ("l_p_hip", 1, (-0.4, 3.3)),
                ("l_r_hip", -1, (-0.4, 0.7)),
            ],
        },
        "r_hip": {
            "seq": "ZXY",  # r_y_hip(Z,flip) -> r_p_hip(X,flip) -> r_r_hip(Y,flip)
            "dofs": [
                ("r_y_hip", -1, (-0.1, 0.6)),
                ("r_p_hip", -1, (-0.4, 3.3)),
                ("r_r_hip", -1, (-0.4, 0.7)),
            ],
        },
    }

    # Indices of the 4 binary foot-contact labels in the 263-dim vector.
    FOOT_CONTACT_START = 256

    # You will need to tune these based on exact URDF/SMPL axis differences.
    # These are 90-degree pitch/roll rotations to map SMPL's bone axis to URDF's -Z axis.
    FRAME_OFFSETS: ClassVar[dict[str, np.ndarray]] = {
        "l_shoulder": Rotation.from_euler("y", 90, degrees=True).as_matrix(),
        "r_shoulder": Rotation.from_euler("y", -90, degrees=True).as_matrix(),
        "l_hip": Rotation.from_euler("x", -90, degrees=True).as_matrix(),
        "r_hip": Rotation.from_euler("x", -90, degrees=True).as_matrix(),
        # Default to identity matrix for joints that map cleanly
    }

    def __init__(
        self,
        anim: np.ndarray,
        fps: float = 20.0,
        initial_xz: tuple[float, float] = (0.0, 0.0),
        initial_yaw: float = 0.0,
        animation_state_fn=None,
    ) -> None:
        """
        :param anim: np.ndarray of shape (T, 263), HumanML3D representation.
        :param fps: frame rate used to compute the "t" field (seconds).
        :param initial_xz: (x0, z0) world-frame offset applied to the
            reconstructed root trajectory, i.e. where frame 0 is placed.
        :param initial_yaw: world-frame heading (radians) applied to the
            reconstructed root trajectory, i.e. which way frame 0 faces.
        :param animation_state_fn: optional callable
            (frame_idx: int, foot_contacts: np.ndarray[4]) -> int, used to
            populate "animation_state". Defaults to packing the 4 binary
            foot-contact labels into a 4-bit int (bit0=fc0 ... bit3=fc3).
            Foot-contact semantics aren't specified anywhere else in the
            format, so treat the default as a placeholder to override.
        """
        self.set_source_anim(anim)
        self.fps = fps
        self.initial_xz = np.asarray(initial_xz, dtype=float)
        self.initial_yaw = initial_yaw
        self.animation_state_fn = animation_state_fn or self._default_animation_state

    def set_source_anim(self, anim: np.ndarray) -> None:
        assert anim.ndim == 2 and anim.shape[1] == 263, (
            f"expected (T, 263), got {anim.shape}"
        )
        self.anim = anim
        self.T = anim.shape[0]

    # ---------------------------------------------------------------- #
    # 6D rotation -> rotation matrix (batched over time)
    # ---------------------------------------------------------------- #
    @staticmethod
    def _rot6d_to_matrix(x: np.ndarray) -> np.ndarray:
        """
        Gram-Schmidt orthogonalization (Zhou et al., 6D continuous rotation
        representation). x: (..., 6) -> (..., 3, 3)
        """
        a1, a2 = x[..., 0:3], x[..., 3:6]
        b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
        proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
        b2 = a2 - proj
        b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
        b3 = np.cross(b1, b2)
        return np.stack([b1, b2, b3], axis=-1)  # columns b1, b2, b3

    def _all_local_rotmats(self) -> np.ndarray:
        """
        Precompute local rotation matrices for every HumanML3D joint (0..21)
        across ALL frames in one vectorized pass. Shape: (T, 22, 3, 3).
        Joint 0 (root) has no stored rotation -> identity placeholder
        (never used directly, only via ROOT_PROXY_JOINT).
        """
        mats = np.tile(np.eye(3), (self.T, 22, 1, 1))
        for j in range(1, 22):
            s = self.ROT_START + (j - 1) * 6
            d6 = self.anim[:, s : s + 6]
            mats[:, j] = self._rot6d_to_matrix(d6)
        return mats

    # ---------------------------------------------------------------- #
    # Kinematic-chain composition
    # ---------------------------------------------------------------- #
    def _chain_between(self, child: int, ancestor: int) -> list[int]:
        """
        Walk up HML3D_PARENTS from `child` to `ancestor` (exclusive),
        returning the joint ids in ancestor-to-child order.
        """
        chain = []
        j = child
        while j != ancestor:
            if j == -1:
                raise ValueError(f"ancestor {ancestor} not found above joint {child}")
            chain.append(j)
            j = self.HML3D_PARENTS[j]
        chain.reverse()
        return chain

    @staticmethod
    def _composed_rotation(local_mats: np.ndarray, chain: list[int]) -> np.ndarray:
        """
        Compose local rotations along `chain` (ancestor-side first) to get
        the rotation of the chain's end joint relative to the ancestor,
        for all frames at once. Returns (T, 3, 3).
        """
        t = local_mats.shape[0]
        rel = np.tile(np.eye(3), (t, 1, 1))
        for j in chain:
            rel = np.einsum("tij,tjk->tik", rel, local_mats[:, j])
        return rel

    # ---------------------------------------------------------------- #
    # Angle extraction
    # ---------------------------------------------------------------- #
    @staticmethod
    def _extract_twist_angle(
        rel: np.ndarray, axis: tuple[float, float, float]
    ) -> np.ndarray:
        """
        Swing-twist decomposition: extract the rotation angle about `axis`,
        discarding any off-axis component. Vectorized over T. Returns (T,).
        """
        axis = np.asarray(axis, dtype=float)
        quat = Rotation.from_matrix(rel).as_quat()  # (T, 4) = x,y,z,w
        proj = quat[:, 0] * axis[0] + quat[:, 1] * axis[1] + quat[:, 2] * axis[2]
        w = quat[:, 3]
        return 2.0 * np.arctan2(proj, w)

    # ---------------------------------------------------------------- #
    # Root trajectory reconstruction (angular vel + linear vel -> pose)
    # ---------------------------------------------------------------- #
    @staticmethod
    def _qinv(q: np.ndarray) -> np.ndarray:
        """Conjugate of a unit quaternion, (..., 4) = (w, x, y, z)."""
        q = q.copy()
        q[..., 1:] *= -1
        return q

    @staticmethod
    def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate vectors v (..., 3) by quaternions q (..., 4) = (w, x, y, z)."""
        qvec = q[..., 1:]
        uv = np.cross(qvec, v)
        uuv = np.cross(qvec, uv)
        return v + 2.0 * (q[..., :1] * uv + uuv)

    def _recover_root_trajectory(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct root yaw and XZ position by integrating:
          - anim[:, 0]: root angular (yaw) velocity
          - anim[:, 1:3]: root linear velocity in X/Z, expressed in the
            root's *local* (facing-relative) frame at that timestep
        following the standard HumanML3D root-recovery convention: the
        yaw at frame t is the cumulative sum of *previous* frames' angular
        velocity (frame 0 starts at yaw 0), and each frame's local-frame
        XZ velocity is rotated into the world frame using the inverse of
        the yaw *at the time it was measured* before being accumulated.

        Returns (yaw: (T,), xz: (T, 2)), both already offset by
        `initial_yaw` / `initial_xz`.
        """
        rot_vel = self.anim[:, 0]
        yaw = np.zeros(self.T)
        yaw[1:] = rot_vel[:-1]
        yaw = np.cumsum(yaw)

        # Quaternion (w, x, y, z) for a rotation about the Y (up) axis.
        r_rot_quat = np.zeros((self.T, 4))
        r_rot_quat[:, 0] = np.cos(yaw)
        r_rot_quat[:, 2] = np.sin(yaw)

        local_vel = np.zeros((self.T, 3))
        local_vel[1:, [0, 2]] = self.anim[:-1, 1:3]

        world_vel = self._qrot(self._qinv(r_rot_quat), local_vel)
        # FIX: Map SMPL (Y-up) to ROS (Z-up)
        # SMPL Z (forward) -> ROS X
        # SMPL X (right)   -> ROS -Y
        ros_x = np.cumsum(world_vel[:, 2], axis=0)
        ros_y = np.cumsum(-world_vel[:, 0], axis=0)

        xz = np.stack([ros_x, ros_y], axis=1)  # Note: 'xz' variable now holds (X, Y)

        # Apply the assumed initial pose (where/which way frame 0 starts).
        cos0, sin0 = np.cos(self.initial_yaw), np.sin(self.initial_yaw)
        rot2d = np.array([[cos0, sin0], [-sin0, cos0]])
        xz = xz @ rot2d.T + self.initial_xz
        yaw = yaw + self.initial_yaw

        return yaw, xz

    # ---------------------------------------------------------------- #
    # Animation state (placeholder: packs foot-contact bits)
    # ---------------------------------------------------------------- #
    @staticmethod
    def _default_animation_state(frame_idx: int, foot_contacts: np.ndarray) -> int:
        bits = (foot_contacts > 0.5).astype(int)
        return int(bits[0] | (bits[1] << 1) | (bits[2] << 2) | (bits[3] << 3))

    def _compute_angles(self) -> dict[str, np.ndarray]:
        """All 20 ROS4HRI DOFs, vectorized over T. dict[name] -> (T,) array."""
        local = self._all_local_rotmats()  # (T, 22, 3, 3)
        results: dict[str, np.ndarray] = {}

        for group, joint_id in self.HML3D_ROS4HRI_MAP.items():
            if group == "torso":
                continue  # fixed joint, no DOF in the ROS4HRI tree

            if group == "waist":
                chain = self._chain_between(self.ROOT_PROXY_JOINT, 0)
            else:
                ancestor = self.ROS4HRI_ANCESTOR[joint_id]
                chain = self._chain_between(joint_id, ancestor)

            rel = self._composed_rotation(local, chain)

            # FIX: Apply local frame offsets
            offset = self.FRAME_OFFSETS.get(group, np.eye(3))
            offset_inv = np.linalg.inv(offset)
            # Vectorized matrix multiplication: Offset * Rel * Offset_inv
            rel_aligned = np.einsum("ij,tjk->tik", offset, rel)
            rel_aligned = np.einsum("tij,jk->tik", rel_aligned, offset_inv)

            if group in self.SINGLE_DOF_SPEC:
                name, axis, limits = self.SINGLE_DOF_SPEC[group]
                # Use rel_aligned instead of rel
                angle = self._extract_twist_angle(rel_aligned, axis)
                results[name] = np.clip(angle, limits[0], limits[1])
            else:
                spec = self.MULTI_DOF_SPEC[group]
                # Use rel_aligned instead of rel
                euler = Rotation.from_matrix(rel_aligned).as_euler(
                    spec["seq"], degrees=False
                )
                for i, (name, sign, limits) in enumerate(spec["dofs"]):
                    angle = euler[:, i] * sign
                    results[name] = np.clip(angle, limits[0], limits[1])

        return results

    def convert(self) -> list[dict]:
        """
        Returns a list (length T) of per-frame dicts:
            {
                "angles": {JOINT_NAME: float, ...},   # len 20
                "root_xz_yaw": (x, z, yaw),
                "animation_state": int,
                "t": float,
            }
        """
        angle_results = self._compute_angles()
        names = list(angle_results.keys())
        angle_stack = np.stack([angle_results[n] for n in names], axis=1)  # (T, 20)

        yaw, xz = self._recover_root_trajectory()  # (T,), (T, 2)
        foot_contacts = self.anim[
            :, self.FOOT_CONTACT_START : self.FOOT_CONTACT_START + 4
        ]
        times = np.arange(self.T) / self.fps

        frames = []
        for t in range(self.T):
            frames.append(
                {
                    "angles": dict(zip(names, angle_stack[t].tolist())),
                    "root_xy_yaw": (float(xz[t, 0]), float(xz[t, 1]), float(yaw[t])),
                    "animation_state": self.animation_state_fn(t, foot_contacts[t]),
                    "t": float(times[t]),
                }
            )
        return frames


if __name__ == "__main__":
    # Smoke test with random data
    rng = np.random.default_rng(0)
    T = 50
    source_anim = np.load("sample0_repeat0_len164_263.npy")

    # foot-contact channels are meant to be binary-ish; clip so the
    # placeholder animation_state function does something sane
    # fake_anim[:, 256:260] = rng.integers(0, 2, size=(T, 4))

    conv = HumanML3D2ROS4HRIConverter(source_anim, fps=20.0)
    frames = conv.convert()
    print(f"Converted {len(frames)} frames, {len(frames[0]['angles'])} DOFs each")
    f0, f1 = frames[0], frames[1]
    print(
        "Frame 0:",
        {
            k: (round(v, 3) if isinstance(v, float) else v)
            for k, v in f0.items()
            if k != "angles"
        },
    )
    print("  angles:", {k: round(v, 3) for k, v in f0["angles"].items()})
    print("Frame 1 root_xy_yaw:", tuple(round(v, 4) for v in f1["root_xy_yaw"]))
    np.save("converted_anim.npy", frames)
