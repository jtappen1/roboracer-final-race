#!/usr/bin/env python3
"""
lane_switcher_node.py

Behavior:
  - ALWAYS follow the center (optimal) lane by default.
  - If an obstacle is detected within blocking_radius of ANY waypoint on the
    center lane within lookahead_window_dist ahead of the robot, switch to the
    best unblocked alternative (left or right).
  - Only return to center after the center lane has been continuously clear
    for clear_confirm_count consecutive obstacle callbacks (hysteresis).

Subscribes:
  - geometry_msgs/PoseArray  <- /obstacles/centroids
  - nav_msgs/Odometry        <- /ego_racecar/odom

Publishes:
  - geometry_msgs/PoseArray        -> /planning/active_lane
  - std_msgs/String                -> /planning/active_lane_name
  - visualization_msgs/MarkerArray -> /planning/lane_markers
"""

import csv
import math

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, Pose, PoseArray, Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import ColorRGBA, Header, String
from visualization_msgs.msg import Marker, MarkerArray


LANE_COLORS = {
    'center': ColorRGBA(r=0.2, g=0.8, b=0.2, a=1.0),  # green
    'left':   ColorRGBA(r=0.2, g=0.6, b=1.0, a=1.0),  # blue
    'right':  ColorRGBA(r=1.0, g=0.8, b=0.1, a=1.0),  # yellow
}


class LaneSwitcherNode(Node):

    def __init__(self):
        super().__init__('lane_switcher_node')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('waypoints_csv',
            '/home/jtappen/roboracer_ws/src/final-race/pure_pursuit/path/levine_2floor_points.csv')
        self.declare_parameter('map_frame',             'map')
        self.declare_parameter('lane_offset',           0.35)   # m, lateral shift per lane
        self.declare_parameter('blocking_radius',       0.3)  # m, obstacle-to-waypoint dist = blocked
        self.declare_parameter('lookahead_window_dist', 3.0)   # m, how far ahead to scan for obstacles
        self.declare_parameter('min_publish_dist',      2.0)   # m, minimum path length published
        self.declare_parameter('clear_confirm_count',   240)    # scans center must be clear before returning
        self.declare_parameter('obstacles_topic',       '/obstacles/centroids')
        self.declare_parameter('odom_topic',            '/ego_racecar/odom')


        waypoints_csv              = self.get_parameter('waypoints_csv').value
        self.map_frame             = self.get_parameter('map_frame').value
        self.lane_offset           = self.get_parameter('lane_offset').value
        self.blocking_radius       = self.get_parameter('blocking_radius').value
        self.lookahead_window_dist = self.get_parameter('lookahead_window_dist').value
        self.min_publish_dist      = self.get_parameter('min_publish_dist').value
        self.clear_confirm_count   = self.get_parameter('clear_confirm_count').value
        obstacles_topic            = self.get_parameter('obstacles_topic').value
        odom_topic                 = self.get_parameter('odom_topic').value

        # ── State ────────────────────────────────────────────────────────────
        self.robot_pos          = np.array([0.0, 0.0])
        self.robot_yaw          = 0.0
        self.active_lane        = 'center'
        self.center_clear_count = 0          # hysteresis: how long center has been clear

        self.obstacle_positions = np.empty((0, 2))

        # ── Build lanes ──────────────────────────────────────────────────────
        center_pts = self._load_waypoints(waypoints_csv)
        if center_pts is None:
            self.get_logger().fatal('Cannot load waypoints — aborting.')
            raise RuntimeError('No waypoints loaded')

        self.lanes = {
            'center': center_pts,
            'left':   self._offset_lane(center_pts, +self.lane_offset),
            'right':  self._offset_lane(center_pts, -self.lane_offset),
        }
        self.get_logger().info(
            f'Lanes ready: {len(center_pts)} pts, offset=±{self.lane_offset}m'
        )

        # ── Subscribers / Publishers ─────────────────────────────────────────
        self.create_subscription(PoseArray, obstacles_topic, self._obstacles_cb, 10)
        self.create_subscription(Odometry,  odom_topic,      self._odom_cb,      10)

        self.lane_pub        = self.create_publisher(PoseArray,   '/planning/active_lane',      10)
        self.lane_name_pub   = self.create_publisher(String,      '/planning/active_lane_name', 10)
        self.lane_marker_pub = self.create_publisher(MarkerArray, '/planning/lane_markers',     10)

        self.create_timer(0.1, self._publish_lane_markers)

        self.get_logger().info(
            f'LaneSwitcherNode ready | '
            f'blocking_radius={self.blocking_radius}m  '
            f'lookahead_window={self.lookahead_window_dist}m  '
            f'clear_hysteresis={self.clear_confirm_count}'
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Subscribers
    # ═══════════════════════════════════════════════════════════════════════

    def _odom_cb(self, msg: Odometry):
        self.robot_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        ])
        q = msg.pose.pose.orientation
        self.robot_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def _obstacles_cb(self, msg: PoseArray):
        # ── 1. Update obstacle list ───────────────────────────────────────
        if msg.poses:
            self.obstacle_positions = np.array(
                [[p.position.x, p.position.y] for p in msg.poses]
            )
        else:
            self.obstacle_positions = np.empty((0, 2))

        # ── 2. Evaluate which lanes are blocked right now ─────────────────
        center_blocked = self._is_lane_blocked('center')
        left_blocked   = self._is_lane_blocked('left')
        right_blocked  = self._is_lane_blocked('right')

        self.get_logger().debug(
            f'blocked → center:{center_blocked} left:{left_blocked} right:{right_blocked} '
            f'| active:{self.active_lane} clear_count:{self.center_clear_count}'
        )

        # ── 3. State machine ──────────────────────────────────────────────
        if self.active_lane == 'center':
            if center_blocked:
                # Leave center — choose the first unblocked side
                new_lane = self._pick_alternative(left_blocked, right_blocked)
                if new_lane != 'center':
                    self.get_logger().info(
                        f'[SWITCH] center blocked → {new_lane}'
                    )
                    self.active_lane        = new_lane
                    self.center_clear_count = 0
                # if both sides also blocked: stay on center (least bad)

        else:
            # ── On left or right ─────────────────────────────────────────
            if not center_blocked:
                # Center is clear — count up toward hysteresis threshold
                self.center_clear_count += 1
                if self.center_clear_count >= self.clear_confirm_count:
                    self.get_logger().info(
                        f'[RETURN] {self.active_lane} → center '
                        f'(clear for {self.center_clear_count} consecutive scans)'
                    )
                    self.active_lane        = 'center'
                    self.center_clear_count = 0
            else:
                # Center still blocked — reset counter, stay off center
                self.center_clear_count = 0

                # Current side lane is also blocked — do NOT swap directly
                # to the other side. Must return to center first.
                cur_blocked = left_blocked if self.active_lane == 'left' else right_blocked
                if cur_blocked:
                    self.get_logger().warn(
                        f'{self.active_lane} lane blocked but cannot swap directly '
                        f'to other side — must return to center first.',
                        throttle_duration_sec=0.5
                    )

        # ── 4. Publish ────────────────────────────────────────────────────
        self._publish_active_lane(msg.header)

    # ═══════════════════════════════════════════════════════════════════════
    # Lane blocking check
    # ═══════════════════════════════════════════════════════════════════════

    def _is_lane_blocked(self, lane_name: str) -> bool:
        """
        True if any obstacle centroid is within blocking_radius of ANY
        waypoint on this lane within lookahead_window_dist ahead of the robot.

        Checks a window of waypoints (not just a single lookahead point) so a
        wide or angled obstacle can't sneak through between waypoints.
        """
        if self.obstacle_positions.shape[0] == 0:
            return False

        window = self._get_lookahead_window(self.lanes[lane_name])  # Mx2
        if window.shape[0] == 0:
            return False

        # Vectorised MxN distance matrix — O(M*N) but M and N are small
        diff  = window[:, np.newaxis, :] - self.obstacle_positions[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)   # MxN
        return bool(dists.min() < self.blocking_radius)

    def _get_lookahead_window(self, lane: np.ndarray) -> np.ndarray:
        """
        Return waypoints on `lane` that are within lookahead_window_dist
        of the robot, starting from the nearest waypoint and walking forward.
        """
        dists_to_robot = np.linalg.norm(lane - self.robot_pos, axis=1)
        nearest_idx    = int(np.argmin(dists_to_robot))
        n              = len(lane)

        window   = []
        acc_dist = 0.0
        prev_pt  = lane[nearest_idx]

        for i in range(n):
            idx = (nearest_idx + i) % n
            pt  = lane[idx]
            acc_dist += np.linalg.norm(pt - prev_pt)
            if acc_dist > self.lookahead_window_dist:
                break
            window.append(pt)
            prev_pt = pt

        return np.array(window) if window else lane[nearest_idx:nearest_idx + 1]

    # ═══════════════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _pick_alternative(self, left_blocked: bool, right_blocked: bool) -> str:
        """Return the first unblocked side lane, or 'center' if both are blocked."""
        if not left_blocked:
            return 'left'
        if not right_blocked:
            return 'right'
        self.get_logger().warn(
            'All lanes blocked — holding center.',
            throttle_duration_sec=1.0
        )
        return 'center'

    def _offset_lane(self, waypoints: np.ndarray, offset: float) -> np.ndarray:
        """Shift waypoints laterally by offset metres (+ = left, - = right)."""
        n      = len(waypoints)
        result = np.empty_like(waypoints)
        for i in range(n):
            fwd     = waypoints[(i + 1) % n] - waypoints[(i - 1) % n]
            tangent = fwd / (np.linalg.norm(fwd) + 1e-9)
            normal  = np.array([-tangent[1], tangent[0]])  # 90° CCW = left
            result[i] = waypoints[i] + normal * offset
        return result

    def _load_waypoints(self, csv_path: str):
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
                            pass
            if not pts:
                self.get_logger().error(f'No waypoints found in {csv_path}')
                return None
            return np.array(pts, dtype=np.float64)
        except Exception as e:
            self.get_logger().error(f'Failed to load waypoints: {e}')
            return None

    # ═══════════════════════════════════════════════════════════════════════
    # Publishers
    # ═══════════════════════════════════════════════════════════════════════

    def _publish_active_lane(self, header: Header):
        """
        Publish waypoints starting from the nearest point on the active lane,
        walking forward until at least min_publish_dist of path is covered.
        This guarantees pure pursuit always has enough lookahead to work with.
        """
        lane      = self.lanes[self.active_lane]
        n         = len(lane)
        dists     = np.linalg.norm(lane - self.robot_pos, axis=1)
        start_idx = int(np.argmin(dists))

        pts      = []
        acc_dist = 0.0
        prev_pt  = lane[start_idx]

        for i in range(n):
            idx = (start_idx + i) % n
            pt  = lane[idx]
            acc_dist += np.linalg.norm(pt - prev_pt)
            pts.append(pt)
            prev_pt = pt
            if acc_dist >= self.min_publish_dist:
                break

        pa                 = PoseArray()
        pa.header          = header
        pa.header.frame_id = self.map_frame
        for x, y in pts:
            p               = Pose()
            p.position.x    = float(x)
            p.position.y    = float(y)
            p.orientation.w = 1.0
            pa.poses.append(p)
        self.lane_pub.publish(pa)

        name_msg      = String()
        name_msg.data = self.active_lane
        self.lane_name_pub.publish(name_msg)

    def _publish_lane_markers(self):
        """Visualise all three lanes in Foxglove. Active lane is brighter/thicker."""
        ma     = MarkerArray()
        header = Header()
        header.stamp    = self.get_clock().now().to_msg()
        header.frame_id = self.map_frame

        del_all        = Marker()
        del_all.header = header
        del_all.ns     = 'lanes'
        del_all.action = Marker.DELETEALL
        ma.markers.append(del_all)

        for lane_id, (name, pts) in enumerate(self.lanes.items()):
            is_active = (name == self.active_lane)
            color     = LANE_COLORS[name]

            # Full lane line strip
            line        = Marker()
            line.header = header
            line.ns     = 'lane_lines'
            line.id     = lane_id
            line.type   = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale  = Vector3(x=0.06 if is_active else 0.02, y=0.0, z=0.0)
            line.color  = ColorRGBA(
                r=color.r, g=color.g, b=color.b,
                a=1.0 if is_active else 0.3
            )
            for x, y in pts:
                p = Point(); p.x = float(x); p.y = float(y); p.z = 0.0
                line.points.append(p)
            # Close loop
            if len(pts):
                p = Point(); p.x = float(pts[0][0]); p.y = float(pts[0][1]); p.z = 0.0
                line.points.append(p)
            ma.markers.append(line)

            # Sphere at nearest waypoint on each lane
            nearest_idx = int(np.argmin(np.linalg.norm(pts - self.robot_pos, axis=1)))
            dot         = Marker()
            dot.header  = header
            dot.ns      = 'lane_nearest'
            dot.id      = lane_id
            dot.type    = Marker.SPHERE
            dot.action  = Marker.ADD
            dot.pose.position.x    = float(pts[nearest_idx][0])
            dot.pose.position.y    = float(pts[nearest_idx][1])
            dot.pose.position.z    = 0.15
            dot.pose.orientation.w = 1.0
            dot.scale  = Vector3(x=0.18, y=0.18, z=0.18)
            dot.color  = ColorRGBA(r=color.r, g=color.g, b=color.b, a=1.0)
            ma.markers.append(dot)

        self.lane_marker_pub.publish(ma)


# ────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = LaneSwitcherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()