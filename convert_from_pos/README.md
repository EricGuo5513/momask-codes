# Conversion from HumanML3D joint positions representation to ROS4HRI semantic angle format

## Convert principal

Since the two representations do not have the same number of joints and links, we follow this customized mapping:

| # | HumanML3D joint index | ROS4HRI base name | axis | limits [lo, hi] (rad) | role |
|---|---|---|---|---|---|
| 1 | 0 (pelvis) | `waist` | (0,1,0) | [-0.2, 1.0] | torso forward lean |
| 2 | 15 (head) | `r_head` | (1,0,0) | [-1.0, 1.0] | head roll |
| 3 | 15 (head) | `y_head` | (0,0,1) | [-1.4, 1.4] | head yaw |
| 4 | 15 (head) | `p_head` | (0,-1,0) | [-1.5, 1.5] | head pitch |
| 5 | 16 (left shoulder) | `l_y_shoulder` | (0,0,-1) | [-1.1, 1.9] | L shoulder yaw |
| 6 | 16 (left shoulder) | `l_p_shoulder` | (1,0,0) | [-0.4, 3.3] | **L arm abduction (raw axis)** † |
| 7 | 16 (left shoulder) | `l_r_shoulder` | (0,0,1) | [-1.7, 1.5] | L shoulder roll |
| 8 | 18 (left elbow) | `l_elbow` | (0,-1,0) | [0.0, 2.5] | **L elbow** |
| 9 | 17 (right shoulder) | `r_y_shoulder` | (0,0,1) | [-1.1, 1.9] | R shoulder yaw |
| 10 | 17 (right shoulder) | `r_p_shoulder` | (-1,0,0) | [-0.4, 3.3] | **R arm abduction (raw axis)** † |
| 11 | 17 (right shoulder) | `r_r_shoulder` | (0,0,-1) | [-1.7, 1.5] | R shoulder roll |
| 12 | 19 (right elbow) | `r_elbow` | (0,-1,0) | [0.0, 2.5] | **R elbow** |
| 13 | 1 (left hip) | `l_y_hip` | (0,0,-1) | [-0.1, 0.6] | L hip yaw |
| 14 | 1 (left hip) | `l_p_hip` | (1,0,0) | [-0.4, 3.3] | L hip abduction |
| 15 | 1 (left hip) | `l_r_hip` | (0,-1,0) | [-0.4, 0.7] | **L leg sagittal swing** |
| 16 | 4 (left knee) | `l_knee` | (0,-1,0) | [-2.5, 0.0] | **L knee** |
| 17 | 2 (right hip) | `r_y_hip` | (0,0,-1) | [-0.1, 0.6] | R hip yaw |
| 18 | 2 (right hip) | `r_p_hip` | (-1,0,0) | [-0.4, 3.3] | R hip abduction |
| 19 | 2 (right hip) | `r_r_hip` | (0,-1,0) | [-0.4, 0.7] | **R leg sagittal swing** |
| 20 | 5 (right knee) | `r_knee` | (0,-1,0) | [-2.5, 0.0] | **R knee** |
| 21 | 9 (chest) | `torso` | - | - | Torso, remain fixed, but added to complete kinematic tree, notice that in HumanML3D, chest is lower than the shoulders, but in ROS4HRI, the torso is exactly the same height as shoulders |

Input: HumanML3D (22,3) vector giving each joint's 3D position
Output: ROS4HRI 20-Dimension vector, should be in the format of a list (length T) of per-frame dicts:
```python
{
    "angles": {JOINT_NAME: float, ...},   # len 20
    "root_xz_yaw": (x, z, yaw),
    "animation_state": int,
    "t": float,
}:
```

Consider ROS4HRI `body_${id}` as `waist_${id}` where applicable for convenience as they are collocated.
Note: HumanML3D is Y-up, while ROS4HRI is Z-up, X-forward, Y-left.

## ROS4HRI kinematic tree

body_${id} (ROOT LINK)
├── waist_${id} [revolute]
│   └── waist_${id}
│       └── torso_${id} [fixed]
│           └── torso_${id}
│               ├── r_head_${id} [revolute]
│               │   └── r_head_${id}
│               │       └── y_head_${id} [revolute]
│               │           └── y_head_${id}
│               │               └── p_head_${id} [revolute]
│               │                   └── p_head_${id}
│               │                       └── head_${id} [fixed]
│               │                           └── head_${id}
│               │
│               ├── l_y_shoulder_${id} [revolute]  (LEFT ARM)
│               │   └── l_y_shoulder_${id}
│               │       └── l_p_shoulder_${id} [revolute]
│               │           └── l_p_shoulder_${id}
│               │               └── l_r_shoulder_${id} [revolute]
│               │                   └── l_shoulder_${id}
│               │                       └── l_elbow_${id} [revolute]
│               │                           └── l_elbow_${id}
│               │                               └── l_wrist_${id} [fixed]
│               │                                   └── l_wrist_${id}
│               │
│               └── r_y_shoulder_${id} [revolute]  (RIGHT ARM)
│                   └── r_y_shoulder_${id}
│                       └── r_p_shoulder_${id} [revolute]
│                           └── r_p_shoulder_${id}
│                               └── r_r_shoulder_${id} [revolute]
│                                   └── r_shoulder_${id}
│                                       └── r_elbow_${id} [revolute]
│                                           └── r_elbow_${id}
│                                               └── r_wrist_${id} [fixed]
│                                                   └── r_wrist_${id}
│
├── l_y_hip_${id} [revolute]  (LEFT LEG)
│   └── l_y_hip_${id}
│       └── l_p_hip_${id} [revolute]
│           └── l_p_hip_${id}
│               └── l_r_hip_${id} [revolute]
│                   └── l_hip_${id}
│                       └── l_knee_${id} [revolute]
│                           └── l_knee_${id}
│                               └── l_ankle_${id} [fixed]
│                                   └── l_ankle_${id}
│
└── r_y_hip_${id} [revolute]  (RIGHT LEG)
    └── r_y_hip_${id}
        └── r_p_hip_${id}
            └── r_p_hip_${id} [revolute]
                └── r_r_hip_${id} [revolute]
                    └── r_hip_${id}
                        └── r_knee_${id} [revolute]
                            └── r_knee_${id}
                                └── r_ankle_${id} [fixed]
                                    └── r_ankle_${id}

## HumanML3D kinematic tree:
```python
import numpy as np

raw_offsets = np.array([[0,0,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,0,1],
                           [0,0,1],
                           [0,1,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,0,1],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0]])

kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
```

## How to test

Convert the source animation by using
```bash
python humanml3d2ros4hri.py
```

Visualize the converted animation in RViz
```bash
# In the first terminal
ros2 run robot_state_publisher robot_state_publisher human.urdf

# In the second terminal
python visualize_rviz.py

# In the third terminal
rviz2
```
Then choose `world` as Fixed Frame in Global Options, add RobotModel, and choose `/robot_description` as Description Topic.

Compare if the visualized animation matches `sample0_repeat4_len164.mp4`

