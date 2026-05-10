[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/aWlTeX_W)

# ESE 6150 — Final Race (Team 4)

This README shows how the code is organized, how to run it, and how we approached each stage of the problem like capturing the raceline, tracking it, detecting obstacles, and overtaking.

---

## Team Members

| Name | Affiliation | Email | Links |
|------|-------------|-------|-------|
| **John Tappen** | Computer and Information Science, University of Pennsylvania | jtappen@seas.upenn.edu | — |
| **Xiaoqing Zhu** | GRASP Lab, University of Pennsylvania | qing22@seas.upenn.edu | [GitHub](https://github.com/Hiuching-Jyu) · [LinkedIn](https://www.linkedin.com/in/xiaoqing-z-a0407715a/) |
| **Xinyi Wang** | GRASP Lab, University of Pennsylvania | xinyi77@seas.upenn.edu | — |
| **Adithya Raman** | Robotics, University of Pennsylvania | radithya@seas.upenn.edu | — |

---

## Demo

> Race Video: https://youtu.be/tG2q-ZMUy8c?si=EiIFjeKLbBB9Dpjp

---

## TL;DR — What's in this repo

```
final-race-team4-1/
├── final.sh                          # one-shot tmux launcher for the whole race stack
├── COMMAND.md                        # launch commands cheatsheet
├── pure_pursuit/                     # ROS 2 package: planner + tracker + detector
│   ├── src/
│   │   ├── pure_pursuit_zones.cpp    # combined planner + tracker (lane switching + pure pursuit)
│   │   └── object_detector.cpp       # LiDAR gap-based obstacle detector
│   ├── include/nanoflann.hpp         # KD-tree header used for nearest-waypoint lookup
│   ├── path/                         # raceline CSVs (Wpts_optimized_final.csv is the race line)
│   ├── CMakeLists.txt
│   └── package.xml
├── safety_node/                      # AEB-style safety stop
│   └── src/safety_node.cpp
└── raceline_editors/                 # browser tools for drawing/editing waypoints
```

The runtime stack is two C++ ROS 2 nodes plus the f1tenth_stack, particle_filter, and safety_node that come with the car:

- [object_detector.cpp](pure_pursuit/src/object_detector.cpp) — segments the LiDAR scan into discrete obstacles, runs temporal tracking, and publishes confirmed centroids.
- [pure_pursuit_zones.cpp](pure_pursuit/src/pure_pursuit_zones.cpp) — picks the active lane (center / left / right) based on obstacle locations and zone rules, and drives the car along that lane with pure pursuit.

Topics flow:

```
/scan ──► object_detector_node ──► /obstacles/centroids ─┐
                                                         ├─► pure_pursuit_zones ──► /drive
/pf/pose/odom ───────────────────────────────────────────┘
```

---

## Racing strategy

We deliberately chose a conservative strategy: run our best raceline whenever the track is clear, and slow down (~80 % of the nominal speed) whenever an obstacle is detected ahead and switch lane to avoid it. The reasoning is that not crashing* compounds over the rounds, finishing every loop is worth more than the seconds you can win by being aggressive.

So at a high level:

- **No obstacle** → follow the optimized raceline at full speed, with per-waypoint speeds shaped by curvature.
- **Obstacle ahead** → switch to a parallel left/right lane if one is clear; if both sides are also blocked, hold the center and rely on the speed reduction to avoid a rear-end crash.

---

## Building block 1 — Capturing the raceline

We tried two ways of producing waypoints, and ended up using both depending on what we needed.

### a) Optimized raceline from the centerline

For the "fast" lap we used [AhmadAmine998/global_racetrajectory_optimization](https://github.com/AhmadAmine998/global_racetrajectory_optimization). The pipeline is:

1. Build a map (we used `slam_toolbox` — see [COMMAND.md](COMMAND.md) for the launch command).
2. Extract the track centerline and width from the map.
3. Feed it into the optimizer to get a minimum-curvature / minimum-time raceline.
4. The output is a CSV of `(x, y, v, lookahead)` columns — see [Wpts_optimized_final.csv](pure_pursuit/path/Wpts_optimized_final.csv) for the one we raced with.

While this approach can generate smooth and geometry-aware racelines, it has several practical limitations in our setting. First, the optimization process is highly sensitive to track geometry and frequently fails due to normal crossing issues, especially around sharp corners or noisy centerlines. Second, extracting accurate track widths from the occupancy map is tedious and time-consuming, often requiring substantial manual correction and tuning. Finally, although the optimizer outputs speed and lookahead values, these are conservative defaults and do not transfer reliably to the real vehicle; significant hand-tuning is still necessary to achieve stable and competitive on-car performance.

### b) Manual edits with the in-browser raceline editor

Tuning a raceline by re-running the optimizer is slow. We wrote three small HTML tools in [raceline_editors/](raceline_editors/) so we could edit a CSV interactively:

- [waypoint_generator.html](raceline_editors/waypoint_generator.html) — draw a fresh raceline on top of an occupancy grid.
- [waypoint_editor.html](raceline_editors/waypoint_editor.html) — open an existing CSV and drag points around, add or delete them.
- [final_waypoint_editor.html](raceline_editors/final_waypoint_editor.html) — same as above, plus per-point velocity and lookahead editing with a color heatmap.

Open the HTML directly in a browser, drop in a CSV, edit, re-export. This was by far the fastest way to iterate after a bad lap — push the line wider through a corner, drop the speed for two waypoints, save, copy to the car, run again.

---

## Building block 2 — Tracking the raceline with Pure Pursuit

[pure_pursuit_node.py](pure_pursuit/scripts/pure_pursuit_node.py) is a textbook pure-pursuit tracker with two small but important details.

**Forward-search lookahead.** Rather than picking the closest waypoint and going `lookahead` indices forward (which breaks on dense racelines and tight corners), we walk the path forward by *accumulated arc length*. See `compute_lookahead_pt` in [pure_pursuit_node.py:171-206](pure_pursuit/scripts/pure_pursuit_node.py#L171-L206). When the next waypoint would overshoot the lookahead distance, we linearly interpolate between the previous and current waypoint so the goal point is exactly `lookahead` metres ahead in arc length. This makes the tracker robust to non-uniform waypoint spacing.

**The active waypoint set comes from the lane switcher, not a static CSV.** The tracker subscribes to `/planning/active_lane` and overwrites its waypoint buffer every time a new lane is published ([pure_pursuit_node.py:102-106](pure_pursuit/scripts/pure_pursuit_node.py#L102-L106)). So the *same* pure-pursuit logic transparently follows whichever lane the switcher decided is best.

The geometry uses TF (`map → laser`) to project the goal point into the car frame, then the standard pure-pursuit curvature formula `2 * y / L²` with `arctan(wheelbase * curvature)` for the steering command.

---

## Building block 3 — Obstacle detection

[obstacle_detector.py](pure_pursuit/scripts/obstacle_detector.py) processes the 2D LaserScan in three passes:

1. **FOV + range gate.** Keep only points in a forward arc (default ±10 % of the scan) and within `[min_range, max_range]`. Looking sideways at the wall produced too many false positives, so we explicitly do not.
2. **Euclidean gap segmentation.** Walk through neighbouring scan returns and start a new segment whenever the Euclidean gap between two adjacent points exceeds `gap_threshold` (default 15 cm). Each segment is one candidate object. We accept only segments whose point count is in `[min_segment_points, max_segment_points]` — small enough to reject walls, large enough to reject single-point noise. The upper bound is *distance-aware*: at close range we allow more points per segment because a nearby car projects onto more LiDAR rays.
3. **Centerline filter.** Each segment's centroid is transformed into the `map` frame and rejected if it is more than `max_centerline_dist` (default 30 cm) from any waypoint of the active raceline. This kills phantom detections at the edge of the track.

On top of that there is a **temporal filter** ([obstacle_detector.py:204-253](pure_pursuit/scripts/obstacle_detector.py#L204-L253)). We greedily associate centroids frame-to-frame inside `temporal_match_radius`, count `hits` and `misses`, and only publish a track once it has been seen for `min_publish_hits` consecutive frames and is currently visible. This removes one-frame ghosts caused by reflective surfaces or pedestrians at the trackside.

The detector publishes:
- `/obstacles/centroids` (`PoseArray`, `map` frame) — consumed by the lane switcher.
- `/obstacles/markers` (`MarkerArray`) — for Foxglove visualisation.

---

## Building block 4 — Overtaking by lane switching

We precompute three parallel lanes — `center`, `left`, and `right` — and switch between them.



### How a lane is decided "blocked"

For each lane we extract the *lookahead window*, every waypoint within `lookahead_window_dist` ahead of the car, and check the minimum distance from any obstacle centroid to any waypoint in that window. If it's smaller than `blocking_radius`, that lane is blocked. See `_is_lane_blocked` in [lane_switcher.py:191-209](pure_pursuit/scripts/lane_switcher.py#L191-L209). Doing this on a *window* (not just one point) is what stops a wide obstacle from sneaking between waypoints and being missed.

### State machine

Implemented in `_obstacles_cb` in [lane_switcher.py:126-185](pure_pursuit/scripts/lane_switcher.py#L126-L185):

- We default to the center.
- When the center is blocked, we pick the first unblocked side (left preferred, then right). If both sides are also blocked, we stay on center as the least-bad option.
- We only return to center after it has been clear for `clear_confirm_count` consecutive scans (default 30). This **hysteresis** prevents the car from oscillating left/center/left/center as obstacles flicker in and out of view.
- We do **not** allow direct left↔right swaps — you must pass through center. This is intentional: a direct swap would carry the car through the obstacle.

The active lane (a forward-going slice of waypoints) is republished on `/planning/active_lane` after every obstacle callback, so the pure-pursuit node always has a fresh path to follow.

---

## Tuning notes — what we actually changed and why

Things we found ourselves tweaking in nearly every test session:

- **`gap_threshold`** in the detector. Too low → walls split into many fake objects. Too high → a real car merges into the wall. 15 cm worked on our LiDAR.
- **`max_centerline_dist`**. The centerline filter is the main reason we don't see phantom objects on the wall. Tighten it for narrow tracks.
- **`min_publish_hits` / `max_misses`**. The temporal filter is the *other* main reason; raise hits if you see flicker, raise misses if real cars drop in and out.
- **`blocking_radius`** and **`lookahead_window_dist`**. These two control the trade-off between "switching too late" and "switching when there is no real obstacle". Increase the lookahead if you race fast.
- **`clear_confirm_count`**. We bumped this whenever we saw the car flicker between lanes.
- **Per-waypoint velocities**. The raceline editor was the right tool for this — fly through the lap in Foxglove playback, find the point where you understeered, drop the speeds for the 2–3 waypoints leading into that corner, save, retry.
- **Lookahead distance.** Larger lookahead = smoother but cuts corners; smaller = sharper but can oscillate. We ended up with per-waypoint lookahead values stored in the CSV, edited via the editor.

---

## Challenges — what worked and what didn't

### What didn't work: Frenet + Pure Pursuit 

Our [final-project-team4](../final-project-team4) submission used a Frenet-frame planner producing left/right candidate trajectories that pure pursuit then tracked. In simulation it looked great. On the car, it didn't:

- The choice between left and right would flip every few frames because the cost of the two sides was almost identical when the obstacle was close to the centerline, and tiny noise in the obstacle estimate dominated the decision.
- That flipping translated into the car physically swerving left/right/left/right, which at race speed meant either a crash or a complete lock of the steering against the track wall.

We considered adding cost-smoothing, hysteresis on the left/right decision, and a separate planner. In the end we stepped back and asked what the simplest thing was that would not flip — and that was **don't plan, just switch among precomputed lanes** with explicit hysteresis. The lane-switcher above is the result. It's a dumber algorithm, but the dumbness is the point: there's nothing for noise to perturb.

### What didn't work as expected: the optimizer alone

Our first race used the raceline worked last night before race, and we hit the same corner on every second lap, we found that was a localization problem.

To fix it, we pushed the corner waypoints further out from the inside wall and slow the car down through them, all from the browser editor. We didn't have time to re-run the corner on track before the next race, but our first set of laps at race time confirmed the fix.

### Open issue: the detector still produces fake obstacles

The combination of the centerline filter and the temporal filter removed almost all false positives, but not all. Specific failure cases we observed:

- Highly reflective wall sections producing a momentary phantom in the middle of the track.
- People standing close to the boundary occasionally registering as on-track.

Both could probably be killed by a tighter centerline filter once the car is well-localized, plus increasing `min_publish_hits`. We left the parameters loose because we preferred a few false positives over a missed real obstacle.

---

## Results

- We won **3 rounds** outright.
- We lost to the eventual **#1** and **#3** teams.
- Final placing: **#4**.
- In the **last 4 rounds** of the day we ran clean, no crashes on any loop, which is exactly what the conservative strategy was optimizing for.

---

## Future improvements

1. **Spend more time on `global_racetrajectory_optimization`.** We only seriously used the minimum-curvature output. Comparing minimum-time, shortest-path, and minimum-curvature solutions on the same track and learning when each one wins would be worth a full afternoon.
2. **Better detector.** Specifically, replace the segment-size filter with a proper geometric model of "what does an F1TENTH car look like at this distance" so the detector becomes range-aware in a principled way rather than via the current heuristic.
3. **Push the speed envelope.** Our lap was about 80 % of what the car could do. With a cleaner detector and a faster localization update we'd be willing to raise the velocities globally and re-tune corner-by-corner.
4. **Replace the offset-lane trick with a genuine local planner** for cases where the offset lane runs into the wall (very narrow track sections). Today we fall back to "stay on center and slow down" when both sides are blocked; a thin band of dynamic planning would do better.

---

## Build and run

### Prerequisites

- ROS 2 Humble.
- A workspace with `f1tenth_stack`, `particle_filter`, `foxglove_bridge` already built (the F1TENTH bring-up stack on the car).
- This package built in the same workspace:

```bash
cd ~/ros2_ws
colcon build --packages-select final_race_pure_pursuit
source install/setup.bash
```

### One-shot launch with `final.sh`

[final.sh](final.sh) brings up the full race stack in a tmux session with one pane per node:

```bash
chmod +x final.sh
./final.sh
```

The script opens a tmux session called `workspace`, sources both ROS 2 and the workspace overlays, and starts:

| Pane | Command | Purpose |
|------|---------|---------|
| 0 | `ros2 run final_race_pure_pursuit obstacle_detector.py` | LiDAR-based obstacle detection |
| 1 | `ros2 run final_race_pure_pursuit pure_pursuit_node.py` | Raceline tracker |
| 2 | `ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765` | Visualization bridge |
| 3 | `ros2 launch f1tenth_stack sick_bringup_launch.py` | Car drivers (LiDAR / VESC / IMU) |
| 5 | `ros2 launch particle_filter localize_launch.py` | Localization on the prebuilt map |
| 6 | `ros2 run final_race_pure_pursuit lane_switcher.py` | Lane selection state machine |

Detach with `Ctrl+b`, kill the whole stack with `:kill-session`.


### Editing a raceline in the browser

Open any of the HTML files in [raceline_editors/](raceline_editors/) directly in a browser. Drop your CSV onto the page, edit, export, and copy back into [pure_pursuit/path/](pure_pursuit/path/).
