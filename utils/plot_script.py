import math
import numpy as np
import matplotlib
matplotlib.use('Agg', force=True)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3


COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
    [170, 255, 0], [85, 255, 0], [0, 255, 0],
    [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255],
    [85, 0, 255], [170, 0, 255], [255, 0, 255],
    [255, 0, 170], [255, 0, 85]
]


# -----------------------------
# Utils
# -----------------------------
def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) / intervals)
    out = []
    for i in range(bins):
        s = i * intervals
        e = min(s + intervals, len(ll))
        out.append(np.mean(ll[s:e]))
    return out


# -----------------------------
# 2D Pose
# -----------------------------
def plot_2d_pose(pose, pose_tree, class_type, save_path=None, excluded_joints=None):
    fig = plt.figure()
    plt.title(class_type)

    data = np.array(pose, dtype=float)

    if excluded_joints is None:
        plt.scatter(data[:, 0], data[:, 1], color='b', marker='h', s=15)
    else:
        idxs = [i for i in range(data.shape[0]) if i not in excluded_joints]
        plt.scatter(data[idxs, 0], data[idxs, 1], color='b', marker='h', s=15)

    for i, j in pose_tree:
        plt.plot([data[i, 0], data[j, 0]],
                 [data[i, 1], data[j, 1]], color='r', linewidth=2.0)

    if save_path:
        plt.savefig(save_path)

    plt.close()


# -----------------------------
# 3D single pose
# -----------------------------
def plot_3d_pose_v2(savePath, kinematic_tree, joints, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if title:
        ax.set_title(title)

    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='black')

    for chain in kinematic_tree:
        ax.plot3D(joints[chain, 0], joints[chain, 1], joints[chain, 2], linewidth=2)

    plt.show()
    plt.close()


# -----------------------------
# MAIN 3D ANIMATION (FIXED)
# -----------------------------
def plot_3d_motion(save_path, kinematic_tree, joints, title,
                   figsize=(10, 10), fps=20, radius=4):

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    data = np.array(joints).reshape(len(joints), -1, 3)

    # Rotate skeleton 90 degrees around vertical axis
    theta = np.radians(90)

    rotation_matrix = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    data = np.einsum('ij,tbj->tbi', rotation_matrix, data)

    MINS = data.min(axis=(0, 1))
    MAXS = data.max(axis=(0, 1))

    def init():
        ax.set_xlim(-radius/2, radius/2)
        ax.set_ylim(-radius/2, radius/2)
        ax.set_zlim(0, radius)
        fig.suptitle(title)

    def update(frame):
        ax.cla()
        init()

        ax.view_init(elev=120, azim=-90)

        ax.scatter(
            data[frame, :, 0],
            data[frame, :, 1],
            data[frame, :, 2],
            color='black'
        )

        for i, chain in enumerate(kinematic_tree):
            ax.plot3D(
                data[frame, chain, 0],
                data[frame, chain, 1],
                data[frame, chain, 2],
                linewidth=2,
                color='blue'
            )

    ani = FuncAnimation(fig, update, frames=len(data), interval=1000/fps, repeat=False)

    writer = FFMpegWriter(fps=fps)
    ani.save(save_path, writer=writer)

    plt.close()