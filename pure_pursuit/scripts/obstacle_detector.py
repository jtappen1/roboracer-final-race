#!/usr/bin/env python3
"""
lidar_obstacle_node.py

Segments a 2D LaserScan using Euclidean gap detection — if two adjacent
scan points are more than gap_threshold apart, they belong to different
objects. Each segment that passes size and centerline filters is published
as an obstacle centroid and Foxglove sphere.

Publishes:
  - geometry_msgs/PoseArray        -> /obstacles/centroids  (map frame)
  - visualization_msgs/MarkerArray -> /obstacles/markers    (map frame)
"""

import math
import csv

import numpy as np
import rclpy
from rclpy.node import Node

from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header


class LidarObstacleNode(Node):

    def __init__(self):
        super().__init__('lidar_obstacle_node')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('scan_topic',          '/scan')
        self.declare_parameter('map_frame',           'map')
        self.declare_parameter('obstacle_radius',     0.5)    # sphere radius for visualisation (m)

        # Scan filtering
        self.declare_parameter('min_range',           0.0)   # ignore returns closer than this (m)
        self.declare_parameter('max_range',           5.0)   # ignore returns beyond this (m)
        self.declare_parameter('fov_fraction',        0.1)    # fraction of scan arc each side of fwd

        # Gap segmentation
        self.declare_parameter('gap_threshold',       0.2)   # >10cm between adjacent pts = new segment
        self.declare_parameter('min_segment_points',  10)      # ignore tiny noise segments
        self.declare_parameter('max_segment_points',  100)     # ignore large wall segments
        
        # Distance-based threshold parameters
        self.declare_parameter('max_extended_segment_points', 200) # max points allowed at close range
        self.declare_parameter('max_distance',                3.0) # max distance for scaling

        # Centerline filter
        self.declare_parameter('waypoints_csv', '/home/nvidia/ros2_ws/src/roboracer-final-race/pure_pursuit/path/final_race.csv')
        self.declare_parameter('max_centerline_dist', 0.5)    # reject centroids further than this (m)

        scan_topic               = self.get_parameter('scan_topic').value
        self.map_frame           = self.get_parameter('map_frame').value
        self.obs_radius          = self.get_parameter('obstacle_radius').value
        self.min_range           = self.get_parameter('min_range').value
        self.max_range           = self.get_parameter('max_range').value
        self.fov_fraction        = self.get_parameter('fov_fraction').value
        self.gap_threshold       = self.get_parameter('gap_threshold').value
        self.min_seg_pts         = self.get_parameter('min_segment_points').value
        self.max_seg_pts         = self.get_parameter('max_segment_points').value
        self.max_ext_seg_pts     = self.get_parameter('max_extended_segment_points').value
        self.max_distance        = self.get_parameter('max_distance').value
        waypoints_csv            = self.get_parameter('waypoints_csv').value
        self.max_centerline_dist = self.get_parameter('max_centerline_dist').value

        # Load waypoints
        self.waypoints = self._load_waypoints(waypoints_csv)
        if self.waypoints is not None:
            self.get_logger().info(
                f'Loaded {len(self.waypoints)} waypoints, '
                f'centerline filter: ±{self.max_centerline_dist}m'
            )
        else:
            self.get_logger().warn('No waypoints CSV — centerline filter disabled')

        # ── TF ───────────────────────────────────────────────────────────────
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Subscribers / Publishers ─────────────────────────────────────────
        self.scan_sub     = self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)
        self.centroid_pub = self.create_publisher(PoseArray,   '/obstacles/centroids', 10)
        self.marker_pub   = self.create_publisher(MarkerArray, '/obstacles/markers',   10)

        self.get_logger().info(
            f'LidarObstacleNode ready — topic={scan_topic}, '
            f'gap={self.gap_threshold}m, '
            f'seg_pts=[{self.min_seg_pts},{self.max_seg_pts}], '
            f'frame={self.map_frame}'
        )

    # ────────────────────────────────────────────────────────────────────────
    def scan_callback(self, msg: LaserScan):
        # 1. Build filtered (range, angle) arrays in the laser frame
        num   = len(msg.ranges)
        angles = msg.angle_min + np.arange(num, dtype=np.float32) * msg.angle_increment
        ranges = np.array(msg.ranges, dtype=np.float32)

        # Forward FOV gate
        half_arc = (msg.angle_max - msg.angle_min) * self.fov_fraction
        valid = (
            np.isfinite(ranges)
            & (angles >= -half_arc) & (angles <= half_arc)
            & (ranges >= max(msg.range_min, self.min_range))
            & (ranges <= min(msg.range_max, self.max_range))
        )
        r = ranges[valid]
        a = angles[valid]

        if len(r) < self.min_seg_pts:
            self._publish_empty(msg.header.stamp)
            return

        # 2. TF lookup — laser → map
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                msg.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
        except (LookupException, ExtrapolationException) as e:
            self.get_logger().warn(f'TF lookup failed: {e}', throttle_duration_sec=1.0)
            return

        # 3. Convert to XY in laser frame
        x_laser = r * np.cos(a)
        y_laser = r * np.sin(a)

        # 4. Gap segmentation in laser frame
        #    Euclidean distance between consecutive scan points
        dx   = np.diff(x_laser)
        dy   = np.diff(y_laser)
        gaps = np.sqrt(dx*dx + dy*dy)  # distance between adjacent points

        # Split at gaps > threshold — get indices of gap locations
        split_at = np.where(gaps > self.gap_threshold)[0] + 1  # +1 → start of new segment

        # Build list of index slices for each segment
        starts = np.concatenate([[0], split_at])
        ends   = np.concatenate([split_at, [len(r)]])

        # 5. For each segment: size filter → centroid → transform → centerline filter
        centroids = []
        for s, e in zip(starts, ends):
            seg_len = e - s
            if seg_len < self.min_seg_pts:
                continue

            # Centroid in laser frame
            cx_laser = x_laser[s:e].mean()
            cy_laser = y_laser[s:e].mean()

            # Dynamic calculation of max_segment_points based on distance to segment
            distance = math.hypot(cx_laser, cy_laser)
            clamped_distance = max(0.0, min(distance, self.max_distance))
            distance_ratio = 1.0 - (clamped_distance / self.max_distance)
            dynamic_max_seg_pts = self.max_seg_pts + (self.max_ext_seg_pts - self.max_seg_pts) * distance_ratio

            if seg_len > dynamic_max_seg_pts:
                continue

            # Transform centroid to map frame
            cx_map, cy_map = self._transform_point_2d(cx_laser, cy_laser, tf)

            # Centerline filter
            if self.waypoints is not None:
                dists = np.linalg.norm(self.waypoints - np.array([cx_map, cy_map]), axis=1)
                if dists.min() > self.max_centerline_dist:
                    continue

            centroids.append((cx_map, cy_map))

        # 6. Publish
        header          = Header()
        header.stamp    = msg.header.stamp
        header.frame_id = self.map_frame

        self._publish_centroids(header, centroids)
        self._publish_markers(header, centroids)

        self.get_logger().debug(f'Found {len(centroids)} obstacle(s)')

    # ────────────────────────────────────────────────────────────────────────
    def _transform_point_2d(self, x: float, y: float, tf) -> tuple:
        """Transform a single XY point using a TF stamped transform."""
        t = tf.transform.translation
        q = tf.transform.rotation

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw       = math.atan2(siny_cosp, cosy_cosp)

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        mx = cos_y * x - sin_y * y + t.x
        my = sin_y * x + cos_y * y + t.y
        return mx, my

    # ────────────────────────────────────────────────────────────────────────
    def _load_waypoints(self, csv_path: str):
        """Load x,y waypoints from CSV. Returns Nx2 numpy array or None."""
        if not csv_path:
            return None
        try:
            pts = []
            with open(csv_path, 'r') as f:
                for row in csv.reader(f):
                    if len(row) >= 2:
                        try:
                            pts.append([float(row[0]), float(row[1])])
                        except ValueError:
                            pass  # skip header
            if not pts:
                self.get_logger().error(f'No valid waypoints in {csv_path}')
                return None
            return np.array(pts, dtype=np.float64)
        except Exception as e:
            self.get_logger().error(f'Failed to load waypoints: {e}')
            return None

    # ────────────────────────────────────────────────────────────────────────
    def _publish_centroids(self, header: Header, centroids):
        msg = PoseArray()
        msg.header = header
        for cx, cy in centroids:
            p = Pose()
            p.position.x    = float(cx)
            p.position.y    = float(cy)
            p.position.z    = 0.0
            p.orientation.w = 1.0
            msg.poses.append(p)
        self.centroid_pub.publish(msg)

    # ────────────────────────────────────────────────────────────────────────
    def _publish_markers(self, header: Header, centroids):
        ma = MarkerArray()

        # Clear previous frame
        delete_all        = Marker()
        delete_all.header = header
        delete_all.ns     = 'obstacles'
        delete_all.action = Marker.DELETEALL
        ma.markers.append(delete_all)

        lifetime_ns = int(2 * (1.0 / 15.0) * 1e9)  # 2 lidar frames

        for i, (cx, cy) in enumerate(centroids):
            # Solid centroid dot
            dot                      = Marker()
            dot.header               = header
            dot.ns                   = 'obstacle_centroid'
            dot.id                   = i
            dot.type                 = Marker.SPHERE
            dot.action               = Marker.ADD
            dot.pose.position.x      = float(cx)
            dot.pose.position.y      = float(cy)
            dot.pose.position.z      = 0.0
            dot.pose.orientation.w   = 1.0
            dot.scale                = Vector3(x=0.15, y=0.15, z=0.15)
            dot.color                = ColorRGBA(r=1.0, g=0.2, b=0.2, a=1.0)
            dot.lifetime.nanosec     = lifetime_ns
            ma.markers.append(dot)

            # Transparent radius sphere
            rs                       = Marker()
            rs.header                = header
            rs.ns                    = 'obstacle_radius'
            rs.id                    = i
            rs.type                  = Marker.SPHERE
            rs.action                = Marker.ADD
            rs.pose.position.x       = float(cx)
            rs.pose.position.y       = float(cy)
            rs.pose.position.z       = 0.0
            rs.pose.orientation.w    = 1.0
            d                        = self.obs_radius * 2.0
            rs.scale                 = Vector3(x=d, y=d, z=d)
            rs.color                 = ColorRGBA(r=1.0, g=0.4, b=0.0, a=0.2)
            rs.lifetime.nanosec      = lifetime_ns
            ma.markers.append(rs)

        self.marker_pub.publish(ma)

    # ────────────────────────────────────────────────────────────────────────
    def _publish_empty(self, stamp):
        header          = Header()
        header.stamp    = stamp
        header.frame_id = self.map_frame
        self._publish_centroids(header, [])
        self._publish_markers(header, [])


# ────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = LidarObstacleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()