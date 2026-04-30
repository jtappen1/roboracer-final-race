Window 1: Launch teleop

cd f1tenth_ws
source  /opt/ros/humble/setup.bash
source install/setup.bash 
ros2 launch f1tenth_stack sick_bringup_launch.py


Window 2: Launch slam_toolbox
cd f1tenth_ws/
source  /opt/ros/humble/setup.bash
source install/setup.bash 
ros2 launch slam_toolbox online_async_launch.py slam_params_file:=/home/nvidia/f1tenth_ws/src/f1tenth_system/f1tenth_stack/config/f1tenth_online_async.yaml

ros2 service list | grep slam_toolbox

Window 3: Start foxglove
source /opt/ros/humble/setup.bash
cd f1tenth_ws
source install/setup.bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765


Window 4: Particle Filter
cd f1tenth_ws
source  /opt/ros/humble/setup.bash
source install/setup.bash 
ros2 launch particle_filter localize_launch.py


Window 5:  
cd ros2_ws
source  /opt/ros/humble/setup.bash
source install/setup.bash 
ros2 run pure_pursuit pure_pursuit