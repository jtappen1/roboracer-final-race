#!/usr/bin/env python3
"""
PPO inference node for F1Tenth track following.

Subscribes to (reusing existing topics):
  /ego_racecar/odom  — nav_msgs/Odometry
  /scan              — sensor_msgs/LaserScan

Publishes:
  /drive                          — ackermann_msgs/AckermannDriveStamped  (same as pure_pursuit)
  /final_race_ppo/lateral_error   — std_msgs/Float32   (Foxglove debug)
  /final_race_ppo/heading_error   — std_msgs/Float32   (Foxglove debug)
  /final_race_ppo/goal_marker     — visualization_msgs/Marker  (nearest waypoint, Foxglove debug)

Launch:
    ros2 run final_race_pure_pursuit ppo_node
    ros2 run final_race_pure_pursuit ppo_node --ros-args \
        -p policy_path:=path/ppo_policy \
        -p speed:=4.0 \
        -p lookahead_n:=15 \
        -p corner_slow:=0.6 \
        -p min_speed_ratio:=0.4
"""
import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from tf_transformations import euler_from_quaternion
from visualization_msgs.msg import Marker

# Allow importing ppo_env from the same installed directory
sys.path.insert(0, os.path.dirname(__file__))
from ppo_env import (
    LIDAR_BEAMS, MAX_LIDAR_RANGE, MAX_SPEED, MAX_STEER,
    compute_track_errors, load_waypoints,
)


class PPONode(Node):
    def __init__(self):
        super().__init__("final_race_ppo_node")

        # ── parameters ────────────────────────────────────────────────
        self.declare_parameter("policy_path", "path/ppo_policy")
        self.declare_parameter("waypoints_path", "path/waypoints.csv")
        self.declare_parameter("speed", 4.0)
        # lookahead_n: how many waypoints ahead to read path heading
        # (higher = more corner anticipation; try 10-20)
        self.declare_parameter("lookahead_n", 15)
        # corner_slow: fraction of speed to shed at a full U-turn (0 = none, 1 = stop)
        self.declare_parameter("corner_slow", 0.6)
        # min_speed_ratio: floor on the speed fraction (prevents stopping entirely)
        self.declare_parameter("min_speed_ratio", 0.4)

        pkg_share = get_package_share_directory("final_race_pure_pursuit")

        def resolve(p: str) -> str:
            return p if os.path.isabs(p) else os.path.join(pkg_share, p)

        policy_path  = resolve(self.get_parameter("policy_path").value)
        waypoints_path = resolve(self.get_parameter("waypoints_path").value)
        self.speed           = self.get_parameter("speed").value
        self._lookahead_n    = self.get_parameter("lookahead_n").value
        self._corner_slow    = self.get_parameter("corner_slow").value
        self._min_speed_ratio= self.get_parameter("min_speed_ratio").value

        # ── waypoints ─────────────────────────────────────────────────
        self.waypoints = load_waypoints(waypoints_path)
        self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints")

        # ── policy ────────────────────────────────────────────────────
        self.model = None
        zip_path = policy_path if policy_path.endswith(".zip") else policy_path + ".zip"
        try:
            from stable_baselines3 import PPO as SB3PPO
            self.model = SB3PPO.load(zip_path)
            self.get_logger().info(f"Loaded PPO policy from {zip_path}")
        except FileNotFoundError:
            self.get_logger().error(
                f"Policy file not found: {zip_path}\n"
                "Train first with:  python3 scripts/ppo_train.py"
            )
        except ImportError:
            self.get_logger().error(
                "stable-baselines3 not installed. "
                "Run:  pip install stable-baselines3"
            )

        # ── state ─────────────────────────────────────────────────────
        self._scan: np.ndarray | None = None   # (LIDAR_BEAMS,) normalised
        self._pos:  np.ndarray | None = None   # (2,) world frame
        self._yaw:  float | None      = None
        self._spd:  float             = 0.0

        # ── subscribers ───────────────────────────────────────────────
        self.create_subscription(Odometry, "/ego_racecar/odom",
                                 self._odom_cb, 10)
        self.create_subscription(LaserScan, "/scan",
                                 self._scan_cb, 10)

        # ── publishers ────────────────────────────────────────────────
        self._drive_pub    = self.create_publisher(
            AckermannDriveStamped, "/drive", 10)
        self._lat_pub      = self.create_publisher(Float32, "/final_race_ppo/lateral_error", 10)
        self._head_pub     = self.create_publisher(Float32, "/final_race_ppo/heading_error", 10)
        self._goal_pub     = self.create_publisher(Marker, "/final_race_ppo/goal_marker", 10)

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _scan_cb(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=MAX_LIDAR_RANGE, posinf=MAX_LIDAR_RANGE)
        # Evenly downsample to LIDAR_BEAMS
        idx = np.linspace(0, len(ranges) - 1, LIDAR_BEAMS, dtype=int)
        self._scan = np.clip(ranges[idx], 0.0, MAX_LIDAR_RANGE) / MAX_LIDAR_RANGE

    def _odom_cb(self, msg: Odometry):
        ori = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion(
            [ori.x, ori.y, ori.z, ori.w]
        )
        self._pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        ])
        self._yaw = yaw
        self._spd = float(msg.twist.twist.linear.x)

        self._run_policy()

    # ── inference ─────────────────────────────────────────────────────────────

    def _run_policy(self):
        if self.model is None or self._scan is None or self._pos is None:
            return

        lat_err, head_err, wp_idx = compute_track_errors(
            self._pos, self._yaw, self.waypoints
        )

        # ── Fix 1: lookahead heading ──────────────────────────────────
        # The model was trained with heading from nearest_wp → nearest_wp+1.
        # That only shows a corner WHEN the car is already there.
        # Instead, read the path tangent _lookahead_n waypoints ahead so the
        # policy has time to steer before reaching the corner.
        n = len(self.waypoints)
        la_idx      = (wp_idx + self._lookahead_n) % n
        la_next_idx = (la_idx + 1) % n
        la_tangent  = self.waypoints[la_next_idx] - self.waypoints[la_idx]
        la_angle    = np.arctan2(la_tangent[1], la_tangent[0])
        la_head_err = (la_angle - self._yaw + np.pi) % (2 * np.pi) - np.pi

        # ── Fix 2: match training distribution (fake lidar = all ones) ─
        # The model was trained with _fake_lidar() → np.ones(LIDAR_BEAMS).
        # Feeding real lidar at corners injects wall readings the policy
        # has never seen, causing erratic steering exactly where it matters.
        obs = np.concatenate([
            np.ones(LIDAR_BEAMS, dtype=np.float32),          # match training
            [np.clip(lat_err     / 2.0,  -1.0, 1.0)],
            [np.clip(la_head_err / np.pi, -1.0, 1.0)],       # lookahead heading
            [np.clip(self._spd / MAX_SPEED, 0.0, 1.0)],
        ], dtype=np.float32)

        action, _ = self.model.predict(obs, deterministic=True)
        steer = float(np.clip(action[0], -1.0, 1.0)) * MAX_STEER

        # ── Fix 3: corner speed reduction ────────────────────────────
        # Slow down proportionally to how sharp the upcoming corner is.
        # corner_sharpness ∈ [0, π]: 0 = straight, π = U-turn.
        corner_sharpness = abs(la_head_err)
        speed_ratio = max(
            self._min_speed_ratio,
            1.0 - (corner_sharpness / np.pi) * self._corner_slow,
        )
        speed = self.speed * speed_ratio

        # drive command
        drive = AckermannDriveStamped()
        drive.header.stamp = self.get_clock().now().to_msg()
        drive.header.frame_id = "ego_racecar/base_link"
        drive.drive.steering_angle = steer
        drive.drive.speed = speed
        self._drive_pub.publish(drive)

        # ── Foxglove debug ────────────────────────────────────────────
        self._lat_pub.publish(Float32(data=float(lat_err)))
        self._head_pub.publish(Float32(data=float(la_head_err)))  # lookahead heading

        wp = self.waypoints[wp_idx]
        m = Marker()
        m.header.stamp    = self.get_clock().now().to_msg()
        m.header.frame_id = "map"
        m.ns              = "ppo_goal"
        m.id              = 0
        m.type            = Marker.SPHERE
        m.action          = Marker.ADD
        m.pose.position.x = float(wp[0])
        m.pose.position.y = float(wp[1])
        m.pose.position.z = 0.15
        m.scale.x = m.scale.y = m.scale.z = 0.25
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 1.0
        m.color.a = 1.0
        self._goal_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = PPONode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
