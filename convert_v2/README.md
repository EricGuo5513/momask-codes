# convert_v2

Positions-based HumanML3D -> ROS4HRI converter for the Arena human skeleton
joint contract (`JOINTS.md` in Arena's `task_generator`). A trained
MoMask/T2M model emits a `(T, 22, 3)` HumanML3D joint-position sequence,
this module solves it into per-frame semantic joint angles (24 DOFs) plus a
root trajectory, for playback on the ROS4HRI human rig (RViz renderer, Isaac
bone_map). `convert_from_ang/` is a separate, older angle-input pipeline and
its READMEs describe the 263-dim HumanML3D layout incorrectly (see table
below) -- do not use it as a layout reference.

## HumanML3D 263 layout (correct)

| range | width | content |
|---|---|---|
| `[0:4]` | 4 | root: angular velocity (yaw rate), linear velocity (x, z), root height |
| `[4:67]` | 63 | ric: local joint positions relative to root, 21 joints x 3 |
| `[67:193]` | 126 | rot: local joint rotations (6D continuous representation), 21 joints x 6 |
| `[193:259]` | 66 | local_vel: per-joint linear velocity, 22 joints x 3 |
| `[259:263]` | 4 | feet: binary foot-contact flags (heel/toe, left/right) |

This module does not consume the 263-dim vector directly, it consumes the
already-recovered `(T, 22, 3)` joint positions (the `ric`-derived skeleton
plus root), same input contract as `convert_from_pos/`.

## Clip format

```python
{
    "angles": {base_name: float, ...},  # all 24 base names, unsuffixed
    "root_xy_yaw": (x, y, yaw),          # ROS world: x fwd, y left, yaw CCW about +Z
    "animation_state": int,
    "t": float,                          # seconds from clip start, t[0] == 0.0
}
```

`root_xy_yaw` is ROS-ordered (x=forward, y=left). The older converters in
this repo emit `root_xz_yaw` (HumanML3D-ordered, x=lateral, z=forward): the
key differs deliberately, so a consumer expecting one format fails loudly on
the other instead of swapping axes silently.

No clamping to the advisory `LIMITS` table happens by default (`clamp=False`
is the default of `convert_sequence`/`convert_frame`). Pass `clamp=True` to
clamp the emitted angles.

## Files

- `humanml3d2ros4hri.py`: the converter. `convert_sequence(joints, fps=20.0,
  animation_state=0, clamp_output=False)` -> list of clip frame dicts.
  `project_20dof(frames)` reduces a clip to the 20 gait-driven joints
  (drops r_waist/y_waist/l_ankle/r_ankle, zeros y/r shoulder) for A/B
  comparison against the procedural gait pipeline.
- `validate.py`: standalone validator, see below.

## Running validate.py

```
python3 convert_v2/validate.py
```

Loads `../convert_from_pos/sample0_repeat0_len164.npy`, converts it, and
prints: per-joint continuity (max frame-to-frame delta, flags > 0.5 rad),
per-joint fraction of frames outside the advisory limits, elbow/knee sign
checks, a gait antiphase correlation over the detected walking segment, and
a forward-kinematics reconstruction (rigid bone lengths measured from frame
0, driven by the emitted angles + root) with per-joint mean position error
against the source positions. Saves `converted_v2.npy`,
`validation_frames.png` (source vs. FK-reconstruction stick figures at
frames 0/40/80/120/160) and `validation_traj.png` (top-down root path +
heading arrows). Requires numpy, scipy, matplotlib.
