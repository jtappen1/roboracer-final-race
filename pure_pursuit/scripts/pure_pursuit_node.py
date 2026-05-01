#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from ament_index_python.packages import get_package_share_directory
import os
import yaml
import csv
import tf2_ros
from tf2_ros import TransformException
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point
from tf_transformations import euler_from_quaternion
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PoseArray


# TODO CHECK: include needed ROS msg type headers and libraries

class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')
        self.odom_sub = self.create_subscription(
            Odometry,
            "/ego_racecar/odom",
            self.pose_callback,
            10
        )
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            "/drive",
            10
        )
      
        self.waypoints = np.array(self.get_waypoints(
            "/home/jtappen/roboracer_ws/src/final-race/pure_pursuit/path/levine_2floor_points.csv"
        ))

        # Subscribe to active lane — lane_switcher_node will update this
        self.lane_sub = self.create_subscription(
            PoseArray,
            '/planning/active_lane',
            self._active_lane_callback,
            10
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.source_frame = 'map'
        self.target_frame = 'ego_racecar/base_link'

        self.waypoint_marker_pub = self.create_publisher(
            MarkerArray,
            "ego/pure_pursuit/waypoint_markers",
            10
        )

        self.goal_marker_pub = self.create_publisher(
            MarkerArray,
            "ego/pure_pursuit/goal_marker",
            10
        )
        
        # Note: Remove this when not running sim
        self.timer = self.create_timer(0.5, self.publish_waypoints)  # 2 Hz 

        # Parameters
        self.wheelbase = 0.3
        self.lookahead = 1.5
        self.velocity = 3.5 
    
    def publish_waypoints(self):
        marker_array = MarkerArray()
        for i, wp in enumerate(self.waypoints):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = wp[0]
            marker.pose.position.y = wp[1]
            marker.pose.position.z = 0.1
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.id = i
            marker_array.markers.append(marker)
        self.waypoint_marker_pub.publish(marker_array)
    
    def _active_lane_callback(self, msg: PoseArray):
        if msg.poses:
            self.waypoints = np.array(
                [[p.position.x, p.position.y] for p in msg.poses]
            )   

    def publish_goal_marker(self, goal_world: np.ndarray):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "pure_pursuit_goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = float(goal_world[0])
        marker.pose.position.y = float(goal_world[1])
        marker.pose.position.z = 0.2  

        marker.scale.x = 0.35  
        marker.scale.y = 0.35
        marker.scale.z = 0.35

        # Green color for the goal point
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        arr = MarkerArray()
        arr.markers.append(marker)
        self.goal_marker_pub.publish(arr)

    def map_to_base(self, world_coords) -> np.ndarray:
        point_in_map = PointStamped()
        point_in_map.header.frame_id = self.source_frame
        point_in_map.header.stamp = self.get_clock().now().to_msg()
        point_in_map.point.x = world_coords[0]
        point_in_map.point.y = world_coords[1]
        point_in_map.point.z = 0.0

        try:
            trans = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.source_frame,
                rclpy.time.Time()
            )

            point_in_base = do_transform_point(point_in_map, trans)

            x = point_in_base.point.x
            y = point_in_base.point.y
            return np.array([x, y])

        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as ex:
            self.get_logger().warn(f'Could not transform: {ex}')
            return None


    def get_waypoints(self, csv_path) -> list[tuple[float, float]]:  
        waypoints = []    
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                waypoints.append((float(row[0]), float(row[1])))
        print(f"Loaded {len(waypoints)} sets of waypoints")
        return waypoints
    
    def compute_lookahead_pt(self, curr_world, yaw, lookahead):
        # This version uses forward_mask to find the cloest waypoint 
        # then use forward search to find lookahead waypoint.
        
        N = len(self.waypoints)
        if N == 0:
            return None
        
        delta = self.waypoints[:, :2] - curr_world
        forward = delta[:, 0] * np.cos(yaw) + delta[:, 1] * np.sin(yaw)
        
        waypoints_ahead = self.waypoints[forward > 0]
        wp_ahead_idx = np.flatnonzero(forward > 0)
        if len(wp_ahead_idx) == 0:
            start_idx = int(np.argmin(np.linalg.norm(delta, axis=1)))
        else:
            dist_ahead = np.linalg.norm(delta[wp_ahead_idx], axis=1)
            start_idx = int(wp_ahead_idx[np.argmin(dist_ahead)])
        
        acc_dist = 0.0
        prev_wp = self.waypoints[start_idx, :2]

        # Use accumulated dist of waypoints instead of Euclidean dist.
        for i in range(1, N+1):
            idx = (start_idx + i) % N
            curr_wp = self.waypoints[idx, :2]
            seg_dist = np.linalg.norm(curr_wp - prev_wp)
            if acc_dist + seg_dist >= lookahead:
                remain = lookahead - acc_dist
                # Interpolate if lookahead point is in the middle of two waypoints.
                ratio = remain / seg_dist
                interp_pt = prev_wp + ratio * (curr_wp - prev_wp)
                return interp_pt
            acc_dist += seg_dist
            prev_wp = curr_wp
        return self.waypoints[(start_idx + (N // 2)) % N]


    def pose_callback(self, pose_msg):
        orientation = pose_msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        world_coords = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y])
        
        # Find the closest point to us for the velocity calculation
        delta = self.waypoints[:, :2] - world_coords
        dist = np.linalg.norm(delta, axis=1)
        closest_idx = np.argmin(dist)
        
        # Computes goal world and local coordinates
        goal_world = self.compute_lookahead_pt(world_coords, yaw, self.lookahead)
        goal_local = self.map_to_base(goal_world)
        if goal_local is None:
            return 

        # Calculate Steering Angle
        curvature = 2 * goal_local[1] / (self.lookahead**2)
        steering_angle = np.arctan(self.wheelbase * curvature)

        # Clip the max steering angle
        MAX_STEERING = np.radians(80)
        steering_angle = np.clip(steering_angle, -MAX_STEERING, MAX_STEERING)

        self.publish_goal_marker(goal_world)

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = self.velocity

        self.drive_pub.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
