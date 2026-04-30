"""
assign_velocities.py
--------------------
Reads a CSV of waypoints (x, y) and outputs a new CSV with an assigned
velocity for each point based on path curvature.

Usage:
    python assign_velocities.py input.csv output.csv

Input CSV must have columns named 'x' and 'y' (case-insensitive).
"""

import sys
import math
import csv

# =============================================================================
# CONFIGURATION — tune these to match your robot / vehicle
# =============================================================================

SPEED_STRAIGHT   = 4.0   # m/s  — velocity on straight sections
SPEED_CURVE      = 2.5    # m/s  — velocity on tight curves

# Curve detection: a waypoint is "in a curve" when the turning angle
# (angle between the incoming and outgoing path segments) exceeds this
# threshold in degrees.  Lower = more sensitive.
CURVE_ANGLE_THRESHOLD_DEG = 10.0

# Optional: smooth the velocity so it ramps between straight and curve
# zones instead of switching instantly.  Set to 0 to disable.
SMOOTHING_WINDOW = 6   # number of neighbouring points to average over

# =============================================================================


def load_waypoints(filepath: str) -> list[dict]:
    """Load CSV and return list of {'x': float, 'y': float} dicts."""
    waypoints = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        # normalise column names to lowercase
        fieldnames_lower = {k.lower(): k for k in reader.fieldnames}
        if "x" not in fieldnames_lower or "y" not in fieldnames_lower:
            raise ValueError(
                f"CSV must contain 'x' and 'y' columns. Found: {reader.fieldnames}"
            )
        for row in reader:
            waypoints.append({
                "x": float(row[fieldnames_lower["x"]]),
                "y": float(row[fieldnames_lower["y"]]),
            })
    return waypoints


def turning_angle_deg(p1: dict, p2: dict, p3: dict) -> float:
    """
    Return the turning angle at p2 (the middle point) in degrees.
    This is the angle between vectors (p1→p2) and (p2→p3).
    Returns 0 for collinear points.
    """
    v1 = (p2["x"] - p1["x"], p2["y"] - p1["y"])
    v2 = (p3["x"] - p2["x"], p3["y"] - p2["y"])

    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 == 0 or mag2 == 0:
        return 0.0

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))  # clamp for fp safety
    # acos gives 0 when vectors are parallel (straight ahead),
    # and 90 for a right-angle turn — exactly what we want.
    return math.degrees(math.acos(cos_angle))


def compute_raw_velocities(waypoints: list[dict]) -> list[float]:
    """
    Assign SPEED_STRAIGHT or SPEED_CURVE to each waypoint based on the
    turning angle at that point.
    """
    n = len(waypoints)
    velocities = []

    for i in range(n):
        if i == 0 or i == n - 1:
            # endpoints — no angle to compute, use straight speed
            velocities.append(SPEED_STRAIGHT)
        else:
            angle = turning_angle_deg(waypoints[i - 1], waypoints[i], waypoints[i + 1])
            if angle >= CURVE_ANGLE_THRESHOLD_DEG:
                velocities.append(SPEED_CURVE)
            else:
                velocities.append(SPEED_STRAIGHT)

    return velocities


def smooth_velocities(velocities: list[float], window: int) -> list[float]:
    """
    Look-ahead smoothing: slow down *before* a curve by taking the minimum
    over the next `window` points.  This makes the robot decelerate in
    advance without dragging down distant straight sections.
    """
    if window <= 1:
        return velocities

    n = len(velocities)
    smoothed = []

    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)

        smoothed.append(min(velocities[lo:hi]))

    return smoothed


def write_output(filepath: str, waypoints: list[dict], velocities: list[float]) -> None:
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "velocity"])
        for wp, v in zip(waypoints, velocities):
            writer.writerow([wp["x"], wp["y"], round(v, 4)])


def main():
    if len(sys.argv) < 3:
        print("Usage: python assign_velocities.py <input.csv> <output.csv>")
        sys.exit(1)

    input_file  = sys.argv[1]
    output_file = sys.argv[2]

    print(f"Loading waypoints from: {input_file}")
    waypoints = load_waypoints(input_file)
    print(f"  {len(waypoints)} waypoints loaded.")

    raw_velocities = compute_raw_velocities(waypoints)

    curve_count = sum(1 for v in raw_velocities if v == SPEED_CURVE)
    print(f"  Curve points detected: {curve_count} "
          f"(threshold = {CURVE_ANGLE_THRESHOLD_DEG}°)")

    if SMOOTHING_WINDOW > 1:
        velocities = smooth_velocities(raw_velocities, SMOOTHING_WINDOW)
        print(f"  Smoothing applied (window = {SMOOTHING_WINDOW}).")
    else:
        velocities = raw_velocities

    write_output(output_file, waypoints, velocities)
    print(f"Output written to: {output_file}")


if __name__ == "__main__":
    main()
