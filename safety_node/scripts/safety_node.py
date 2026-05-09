#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import time
import numpy as np
# TODO: include needed ROS msg type headers and libraries
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from rclpy.qos import qos_profile_sensor_data

class SafetyNode(Node):
    """
    The class that handles emergency braking.
    """
    def __init__(self):
        super().__init__('safety_node')
        """
        One publisher should publish to the /drive topic with a AckermannDriveStamped drive message.

        You should also subscribe to the /scan topic to get the LaserScan messages and
        the /ego_racecar/odom topic to get the current speed of the vehicle.

        The subscribers should use the provided odom_callback and scan_callback as callback methods

        NOTE that the x component of the linear velocity in odom is the speed
        """
        self.speed = 0.
        
        # TODO: create ROS subscribers and publishers.
        self.cb_times = []
        self.max_samples = 100
        self.declare_parameter('ttc_threshold', 1.0)
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped, 
            '/drive', 
            10)
        
        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data)
        
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.odom_callback,
            10)
           
        

    def odom_callback(self, odom_msg):
        # TODO: update current speed
        self.speed = odom_msg.twist.twist.linear.x

    def scan_callback(self, scan_msg):
        # TODO: calculate TTC
        start = time.perf_counter()
        # 1. Extract ranges and angles from the scan_msg
        ranges = np.array(scan_msg.ranges, dtype=np.float32)
        # calculate the angles for each beam: angle_i = angle_min + i * angle_increment
        # angles = scan_msg.angle_min + np.arange(len(ranges)) * scan_msg.angle_increment
        angles = scan_msg.angle_min + np.arange(len(ranges), dtype=np.float32) * scan_msg.angle_increment

        # 2. Calculate r_dot
        r_dot = - self.speed * np.cos(angles)

        # 3. Calculate TTC 
        ttc_array = np.full_like(ranges, np.inf, dtype=np.float32)
        valid = np.isfinite(ranges) & (ranges > 0.0)
        approaching = valid & (r_dot < 0.0)

        ttc_array[approaching] = ranges[approaching] / (-r_dot[approaching])

        # 4. Threshold TTC and decide whether to brake
        ttc_threshold = self.get_parameter('ttc_threshold').get_parameter_value().double_value
        min_ttc = np.min(ttc_array)
        # TODO: publish command to brake
        if self.speed > 0.1 and min_ttc < ttc_threshold:
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.drive.speed = 0.0
            drive_msg.drive.steering_angle = 0.0
            # drive_msg.drive.acceleration = -5.0  
            self.drive_publisher.publish(drive_msg)
            self.get_logger().warn(f"Emergency braking triggered. TTC: {min_ttc:.2f}s")
        
        # Calculate the execution time, then compare with C++ version
        end = time.perf_counter()
        dt_ms = (end - start) * 1000.0
        self.cb_times.append(dt_ms)

        if len(self.cb_times) >= self.max_samples:
            avg = sum(self.cb_times) / len(self.cb_times)
            worst = max(self.cb_times)
            self.get_logger().info(
                f"[scan_callback] avg={avg:.3f} ms, worst={worst:.3f} ms"
            )
            self.cb_times.clear()
def main(args=None):
    rclpy.init(args=args)
    safety_node = SafetyNode()
    rclpy.spin(safety_node)
    safety_node.get_logger().info("Shutting down safety node.")
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    safety_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
