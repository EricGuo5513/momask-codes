import numpy as np
from scipy.spatial.transform import Rotation as R


def cont6d_to_matrix(d6):
    """Converts continuous 6D rotation vector to a 3x3 Rotation Matrix."""
    x_raw = d6[:3]
    y_raw = d6[3:]
    x = x_raw / np.linalg.norm(x_raw)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1)


def humanml3d_to_ros4hri(humanml_263_seq, dt=0.05):
    """
    Converts a sequence of HumanML3D 263D features into ROS4HRI joint dicts.
    """
    ros4hri_frames = []

    # HumanML3D to ROS4HRI Coordinate Transform Matrix (Y-up to Z-up, X-forward)
    R_world = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])

    root_x, root_z, root_yaw = 0.0, 0.0, 0.0

    for frame_idx, vec in enumerate(humanml_263_seq):
        # 1. Update Root Pos/Yaw from Root Vel
        r_vel_y = vec[0]
        r_vel_xz = vec[1:3]

        root_yaw += r_vel_y * dt
        root_x += r_vel_xz[0] * dt
        root_z += r_vel_xz[1] * dt

        # 2. Extract 6D Joint Rotations for all 22 joints (0 to 21)
        rot_6d = vec[130:256].reshape(21, 6)

        # Convert ALL relevant joints to 3x3 matrices (including 12 for chest)
        # Note: 0: Pelvis, 1: L_Hip, 2: R_Hip, 4: L_Knee, 5: R_Knee,
        #       9: Spine, 12: Chest, 15: Head, 16: L_Shoulder, 17: R_Shoulder, 18: L_Elbow, 19: R_Elbow
        rot_matrices = {j_idx: cont6d_to_matrix(rot_6d[j_idx]) for j_idx in range(21)}

        # 3. Kinematic Chain Accumulation for Collapsed Chest
        R_pelvis = rot_matrices[0]
        R_spine = rot_matrices[9]
        R_chest = rot_matrices[12]

        # Accumulate rotations down the torso chain: Pelvis -> Spine -> Chest
        R_torso_accumulated = R_spine @ R_chest

        # Fold intermediate torso bending into the shoulder joint orientations
        R_l_shoulder_collapsed = R_torso_accumulated @ rot_matrices[16]
        R_r_shoulder_collapsed = R_torso_accumulated @ rot_matrices[17]

        # 4. Convert Global Pelvis and Local Rotations to Euler Angles
        # Global Pelvis / Waist
        r_pelvis_world = R_world @ R_pelvis
        euler_pelvis = R.from_matrix(r_pelvis_world).as_euler("zyx", degrees=False)

        # Head / Neck
        euler_head = R.from_matrix(rot_matrices[15]).as_euler("xyz", degrees=False)

        # Shoulders (Folded)
        euler_l_shoulder = R.from_matrix(R_l_shoulder_collapsed).as_euler(
            "zyx", degrees=False
        )
        euler_r_shoulder = R.from_matrix(R_r_shoulder_collapsed).as_euler(
            "zyx", degrees=False
        )

        # Elbows & Knees (Local 1DoF Hinge Rotations)
        euler_l_elbow = R.from_matrix(rot_matrices[18]).as_euler("xyz", degrees=False)
        euler_r_elbow = R.from_matrix(rot_matrices[19]).as_euler("xyz", degrees=False)

        euler_l_hip = R.from_matrix(rot_matrices[1]).as_euler("zyx", degrees=False)
        euler_r_hip = R.from_matrix(rot_matrices[2]).as_euler("zyx", degrees=False)

        euler_l_knee = R.from_matrix(rot_matrices[4]).as_euler("xyz", degrees=False)
        euler_r_knee = R.from_matrix(rot_matrices[5]).as_euler("xyz", degrees=False)

        # 5. Populate and Clamp Angles to ROS4HRI Limits
        angles = {
            "waist": float(np.clip(euler_pelvis[1], -0.2, 1.0)),
            "r_head": float(np.clip(euler_head[0], -1.0, 1.0)),
            "y_head": float(np.clip(euler_head[2], -1.4, 1.4)),
            "p_head": float(np.clip(-euler_head[1], -1.5, 1.5)),
            "r_y_shoulder": float(np.clip(euler_r_shoulder[0], -1.1, 1.9)),
            "r_p_shoulder": float(np.clip(euler_r_shoulder[1], -0.4, 3.3)),
            "r_r_shoulder": float(np.clip(euler_r_shoulder[2], -1.7, 1.5)),
            "l_y_shoulder": float(np.clip(euler_l_shoulder[0], -1.1, 1.9)),
            "l_p_shoulder": float(np.clip(euler_l_shoulder[1], -0.4, 3.3)),
            "l_r_shoulder": float(np.clip(euler_l_shoulder[2], -1.7, 1.5)),
            # Flip sign for knee/elbow flexion to match ROS positive rotation convention
            "r_elbow": float(np.clip(abs(euler_r_elbow[1]), 0.0, 2.5)),
            "l_elbow": float(np.clip(abs(euler_l_elbow[1]), 0.0, 2.5)),
            "r_y_hip": float(np.clip(euler_r_hip[0], -0.1, 0.6)),
            "r_p_hip": float(np.clip(euler_r_hip[1], -0.4, 3.3)),
            "r_r_hip": float(np.clip(euler_r_hip[2], -0.4, 0.7)),
            "l_y_hip": float(np.clip(euler_l_hip[0], -0.1, 0.6)),
            "l_p_hip": float(np.clip(euler_l_hip[1], -0.4, 3.3)),
            "l_r_hip": float(np.clip(euler_l_hip[2], -0.4, 0.7)),
            "r_knee": float(np.clip(-euler_r_knee[1], -2.5, 0.0)),
            "l_knee": float(np.clip(-euler_l_knee[1], -2.5, 0.0)),
        }

        ros4hri_frames.append(
            {
                "angles": angles,
                "root_xz_yaw": (float(root_x), float(root_z), float(root_yaw)),
                "animation_state": 1,
                "t": float(frame_idx * dt),
            }
        )

    return ros4hri_frames


if __name__ == "__main__":
    anim = np.load("sample0_repeat0_len164_263.npy")
    frames = humanml3d_to_ros4hri(anim, dt=1 / 20.0)
    np.save("converted_anim.npy", frames)
