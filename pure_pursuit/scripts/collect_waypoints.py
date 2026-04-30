#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
from geometry_msgs.msg import PointStamped
from scipy.interpolate import splprep, splev
import csv
import signal
import sys
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry

class CollectWaypoints(Node):

    def __init__(self):
        super().__init__('collect_waypoints')

        self.clicked_points_sub = self.create_subscription(
            PointStamped,
            "/clicked_point",
            self.clicked_point_callback,
            10
        )
        self.marker_pub = self.create_publisher(
            Marker, 
            "/waypoint_markers", 
            10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            "/pf/pose/odom",
            self.odom_callback,
            10
        )

        self.points = []
        self.marker_id = 0
        self.get_logger().info("Collect Waypoints node started.  Click points to collect, csv made on ctrl c.")

        signal.signal(signal.SIGINT, self.on_shutdown)

        self.latest_odom = None
        self.timer = self.create_timer(0.2, self.process_odom)


    def odom_callback(self, odom_msg):
        self.latest_odom = odom_msg


    def process_odom(self):
        if self.latest_odom is None:
            return

        x = self.latest_odom.pose.pose.position.x
        y = self.latest_odom.pose.pose.position.y

        self.points.append((x, y))
        self.get_logger().info(f"Collected point: ({x:.2f}, {y:.2f})")

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "clicked_points"
        marker.id = self.marker_id
        self.marker_id += 1

        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.1
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.marker_pub.publish(marker)

    
    def clicked_point_callback(self, point_msg):
        x = point_msg.point.x
        y = point_msg.point.y
        self.points.append((x, y))
        self.get_logger().info(f"Collected point: ({x:.2f}, {y:.2f})")

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "clicked_points"
        marker.id = self.marker_id
        self.marker_id += 1

        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.1
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.marker_pub.publish(marker)
    
    def on_shutdown(self, *args):
        self.get_logger().info("Shutting down. Generating spline...")

        if len(self.points) < 2:
            self.get_logger().warn("Not enough points to generate a spline.")
        else:
            points_np = np.array(self.points)
            x = points_np[:, 0]
            y = points_np[:, 1]

            # tck, u = splprep([x, y], s=0, per=True)
            # u_fine = np.linspace(0, 1, 200)
            # x_fine, y_fine = splev(u_fine, tck)

            with open('spline_points.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                for xi, yi in zip(x, y):
                    writer.writerow([xi, yi])
            self.get_logger().info(f"Spline generated and saved to spline_points.csv")

        rclpy.shutdown()
        sys.exit(0)

def main(args=None):
    rclpy.init(args=args)
    node = CollectWaypoints()
    rclpy.spin(node)

if __name__ == "__main__":
    main()