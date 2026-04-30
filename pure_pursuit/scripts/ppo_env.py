#!/usr/bin/env python3
"""
Gymnasium environment for F1Tenth PPO training.
No ROS dependency — runs offline with a kinematic bicycle model.

Observation (23-dim):
  lidar[20]  (downsampled, normalized to [0,1])
  lateral_error  (normalized, clipped to [-1,1])
  heading_error  (normalized, clipped to [-1,1])
  speed          (normalized)

Action (1-dim): steering angle normalized to [-1, 1]
"""
import csv
import os

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── shared constants (imported by ppo_node.py) ────────────────────────────────
WHEELBASE    = 0.33
MAX_STEER    = np.pi / 6          # 30 deg
MAX_SPEED    = 6.0
DT           = 0.05               # 50 ms sim step
LIDAR_BEAMS  = 20
MAX_LIDAR_RANGE = 10.0


# ── waypoint geometry utilities ───────────────────────────────────────────────

def load_waypoints(csv_path: str) -> np.ndarray:
    """Return (N, 2) array of x,y waypoints from CSV."""
    pts = []
    with open(csv_path) as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                try:
                    pts.append((float(row[0]), float(row[1])))
                except ValueError:
                    pass
    return np.array(pts, dtype=np.float64)


def compute_track_errors(pos: np.ndarray, yaw: float,
                          waypoints: np.ndarray):
    """
    Return (lateral_error, heading_error, nearest_idx).

    lateral_error  : signed perpendicular distance from path centre-line
                     (positive = left of path, negative = right)
    heading_error  : yaw_path − yaw_robot  in (−π, π]
    nearest_idx    : index of the closest waypoint
    """
    dists = np.linalg.norm(waypoints - pos, axis=1)
    idx = int(np.argmin(dists))
    next_idx = (idx + 1) % len(waypoints)

    tangent = waypoints[next_idx] - waypoints[idx]
    path_angle = np.arctan2(tangent[1], tangent[0])

    heading_error = path_angle - yaw
    heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

    # signed lateral: project (pos − wp) onto the left-normal of the tangent
    norm_len = np.linalg.norm(tangent) + 1e-9
    left_normal = np.array([-tangent[1], tangent[0]]) / norm_len
    lateral_error = float(np.dot(pos - waypoints[idx], left_normal))

    return float(lateral_error), float(heading_error), idx


# ── Gymnasium environment ─────────────────────────────────────────────────────

class F1TenthEnv(gym.Env):
    """
    Kinematic bicycle model on the waypoint path.

    The fake lidar returns max-range during training (no wall model).
    The policy therefore learns primarily from lateral/heading error;
    real lidar values are available at inference via ppo_node.py.
    """

    metadata = {"render_modes": []}

    def __init__(self, waypoints_path: str = None, speed: float = 4.0):
        super().__init__()

        if waypoints_path is None:
            waypoints_path = os.path.join(
                os.path.dirname(__file__), "..", "path", "waypoints.csv"
            )
        self.waypoints = load_waypoints(waypoints_path)
        self.speed = speed

        obs_dim = LIDAR_BEAMS + 3   # lidar + lat_err + head_err + speed
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self._pos       = np.zeros(2)
        self._yaw       = 0.0
        self._step      = 0
        self._max_steps = 2000
        self._prev_idx  = 0
        self._progress  = 0

    # ------------------------------------------------------------------

    def _fake_lidar(self) -> np.ndarray:
        """All beams at max range (no wall simulation)."""
        return np.ones(LIDAR_BEAMS, dtype=np.float32)

    def _obs(self):
        lat_err, head_err, idx = compute_track_errors(
            self._pos, self._yaw, self.waypoints
        )
        lidar = self._fake_lidar()
        obs = np.concatenate([
            lidar,
            [np.clip(lat_err  / 2.0,      -1.0, 1.0)],
            [np.clip(head_err / np.pi,     -1.0, 1.0)],
            [np.clip(self.speed / MAX_SPEED, 0.0, 1.0)],
        ], dtype=np.float32)
        return obs, lat_err, head_err, idx

    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # start at a random waypoint with slight perturbation
        idx = int(self.np_random.integers(0, len(self.waypoints)))
        wp      = self.waypoints[idx]
        wp_next = self.waypoints[(idx + 1) % len(self.waypoints)]
        tangent = wp_next - wp
        yaw0    = np.arctan2(tangent[1], tangent[0])

        self._pos  = wp + self.np_random.uniform(-0.1, 0.1, size=2)
        self._yaw  = float(yaw0) + float(self.np_random.uniform(-0.1, 0.1))
        self._step = 0
        self._prev_idx = idx
        self._progress = 0

        obs, _, _, _ = self._obs()
        return obs, {}

    def step(self, action):
        steer = float(np.clip(action[0], -1.0, 1.0)) * MAX_STEER

        # kinematic bicycle update
        beta = np.arctan(0.5 * np.tan(steer))
        self._pos = self._pos + self.speed * DT * np.array([
            np.cos(self._yaw + beta),
            np.sin(self._yaw + beta),
        ])
        self._yaw += (self.speed / WHEELBASE) * np.sin(beta) * DT
        self._yaw  = (self._yaw + np.pi) % (2 * np.pi) - np.pi

        obs, lat_err, head_err, idx = self._obs()

        # forward progress (waypoints advanced along track)
        prog = (idx - self._prev_idx) % len(self.waypoints)
        if prog > len(self.waypoints) // 2:
            prog = 0          # detected backwards motion
        self._progress += prog
        self._prev_idx  = idx

        reward = (
              prog * 0.5
            - abs(lat_err)  * 0.5
            - abs(head_err) * 0.1
        )

        self._step += 1
        terminated = abs(lat_err) > 2.0
        truncated  = self._step >= self._max_steps

        info = {
            "lateral_error":  lat_err,
            "heading_error":  head_err,
            "waypoint_idx":   idx,
            "total_progress": self._progress,
        }
        return obs, reward, terminated, truncated, info
