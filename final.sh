#!/usr/bin/env bash
# tmux_launch.sh — open a new tmux session with 5 panes, each running a command

SESSION="workspace"

# Kill existing session if it exists
tmux kill-session -t "$SESSION" 2>/dev/null

# Create a new session (detached), first window
tmux new-session -d -s "$SESSION" -x 220 -y 50

# ── Layout: 5 panes ──────────────────────────────────────────
#
#  ┌──────────────┬──────────────┐
#  │   Pane 1     │   Pane 2     │
#  ├──────┬───────┼──────────────┤
#  │Pane 3│Pane 4 │   Pane 5     │
#  └──────┴───────┴──────────────┘

# Split the initial pane vertically (left | right)
tmux split-window -h -t "$SESSION"

# Split left side horizontally (top | bottom)
tmux split-window -v -t "$SESSION:0.0"

# Split bottom-left horizontally (creating pane 3 and 4 side by side)
tmux split-window -h -t "$SESSION:0.2"

# Split the right side horizontally (top | bottom on right column)
tmux split-window -v -t "$SESSION:0.1"

tmux split-window -h -t "$SESSION:0.3"

tmux split-window -v -t "$SESSION:0.5"


# ── ROS 2 source command (applied to every pane) ──────────────
ROS_SETUP="source /opt/ros/humble/setup.bash && source install/setup.bash"
ROS_SETUP_1="source /opt/ros/humble/setup.bash && cd .. && cd f1tenth_ws && source install/setup.bash"

# ros2 launch realsense2_camera rs_launch.py depth_module.depth_profile:=640x480x30 rgb_camera.color_profile:=640x480x30 pointcloud.enable:=true align_depth.enable:=true

# ── Commands per pane ─────────────────────────────────────────
tmux send-keys -t "$SESSION:0.6" "$ROS_SETUP && ros2 run final_race_pure_pursuit lane_switcher.py"  Enter
tmux send-keys -t "$SESSION:0.0" "$ROS_SETUP && ros2 run final_race_pure_pursuit obstacle_detector.py" Enter
tmux send-keys -t "$SESSION:0.1" "$ROS_SETUP && ros2 run final_race_pure_pursuit pure_pursuit_node.py" Enter
tmux send-keys -t "$SESSION:0.2" "$ROS_SETUP_1 && ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765" Enter
tmux send-keys -t "$SESSION:0.3" "$ROS_SETUP_1 && ros2 launch f1tenth_stack sick_bringup_launch.py" Enter
tmux send-keys -t "$SESSION:0.5" "$ROS_SETUP_1 && ros2 launch particle_filter localize_launch.py" Enter
# tmux send-keys -t "$SESSION:0.4" "$ROS_SETUP && ros2 run final_project tracker_node" Enter


# CHANGED: old tracker command kept below for reference
# tmux send-keys -t "$SESSION:0.4" "$ROS_SETUP && ros2 run final_project tracker_node" Enter

# CHANGED: old optional static TF example kept commented below
# tmux send-keys -t "$SESSION:0.4" "$ROS_SETUP && ros2 run tf2_ros static_transform_publisher 0.25 0.0 -0.25 0.0 0.0 0.0 cloud camera_link & sleep 3 && ros2 run final_project tracker_node" Enter

# CHANGED: optional version if the new pipeline still needs the static TF
# tmux send-keys -t "$SESSION:0.4" "$ROS_SETUP && ros2 run tf2_ros static_transform_publisher 0.25 0.0 -0.25 0.0 0.0 0.0 cloud camera_link & sleep 3 && ros2 run final_project tracker_node" Enter

# Even out the pane sizes
tmux select-layout -t "$SESSION" tiled

# Attach to the session
tmux attach-session -t "$SESSION"