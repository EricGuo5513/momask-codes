#!/usr/bin/env python3

import os
import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation as R


class HumanAnimationPublisher(Node):
    def __init__(self, npy_path, urdf_path, id_suffix="_default"):
        super().__init__("human_animation_publisher")

        self.id_suffix = id_suffix

        # Load converted animation frames
        if not os.path.exists(npy_path):
            self.get_logger().error(f"Animation file not found: {npy_path}")
            raise FileNotFoundError(npy_path)

        self.get_logger().info(f"Loading animation from: {npy_path}")
        # allow_pickle=True is needed since we saved a list of dicts
        self.frames = np.load(npy_path, allow_pickle=True)
        self.total_frames = len(self.frames)
        self.current_frame = 0

        # ROS 2 Publishers
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Publish URDF to robot_state_publisher topic
        if os.path.exists(urdf_path):
            with open(urdf_path, "r") as f:
                robot_desc = f.read()
            from std_msgs.msg import String

            self.urdf_pub = self.create_publisher(String, "/robot_description", 10)
            # Retain description for late-joining RViz nodes
            msg = String()
            msg.data = robot_desc
            self.urdf_pub.publish(msg)

        # Determine playback rate from dt between frames
        dt = 0.05
        if self.total_frames > 1 and "t" in self.frames[0] and "t" in self.frames[1]:
            dt = self.frames[1]["t"] - self.frames[0]["t"]

        self.get_logger().info(
            f"Loaded {self.total_frames} frames. Playing at dt={dt:.3f}s ({1 / dt:.1f} FPS)"
        )

        # Playback Timer
        self.timer = self.create_timer(dt, self.timer_callback)

    def timer_callback(self):
        frame = self.frames[self.current_frame]
        now = self.get_clock().now().to_msg()

        # 1. Publish Joint States
        joint_state = JointState()
        joint_state.header.stamp = now

        angles_dict = frame["angles"]
        for name, val in angles_dict.items():
            # Append suffix matching joint names in URDF (e.g. waist_default)
            joint_state.name.append(f"{name}{self.id_suffix}")
            joint_state.position.append(float(val))

        self.joint_pub.publish(joint_state)

        # 2. Publish Root Transformation (world -> body_default)
        root_x, root_z, root_yaw = frame["root_xz_yaw"]

        tf_msg = TransformStamped()
        tf_msg.header.stamp = now
        tf_msg.header.frame_id = "world"
        tf_msg.child_frame_id = f"body{self.id_suffix}"

        # In ROS (Z-Up, X-Forward): root_x is forward, root_z is lateral (y in ROS frame)
        tf_msg.transform.translation.x = float(root_x)
        tf_msg.transform.translation.y = float(root_z)
        tf_msg.transform.translation.z = 0.0

        # Convert Yaw scalar to Quaternion
        quat = R.from_euler("z", root_yaw).as_quat()  # [x, y, z, w]
        tf_msg.transform.rotation.x = float(quat[0])
        tf_msg.transform.rotation.y = float(quat[1])
        tf_msg.transform.rotation.z = float(quat[2])
        tf_msg.transform.rotation.w = float(quat[3])

        self.tf_broadcaster.sendTransform(tf_msg)

        # Loop Animation
        self.current_frame = (self.current_frame + 1) % self.total_frames


def main(args=None):
    rclpy.init(args=args)

    NPY_PATH = "/home/linh/ductai_nguyen_ws/arena_jazzy_ws/src/Arena/humansim/arena_humansim/resource/263d.npy"
    URDF_PATH = "/home/linh/ductai_nguyen_ws/human.urdf"

    node = HumanAnimationPublisher(NPY_PATH, URDF_PATH)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
