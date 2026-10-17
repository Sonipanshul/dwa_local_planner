# Custom DWA Local Planner for TurtleBot3 (ROS2 Humble)

A Dynamic Window Approach (DWA) local planner 
It drives a simulated TurtleBot3 to a goal while avoiding obstacles detected by the laser scanner.

## Requirements
- Ubuntu 22.04, ROS2 Humble, rviz2
- Gazebo and TurtleBot3 packages:
```bash
  sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-turtlebot3 \
                   ros-humble-turtlebot3-simulations python3-numpy
```

## Build
```bash
mkdir -p ~/dwa_ws/src && cd ~/dwa_ws/src
git clone https://github.com/<your-username>/dwa-local-planner-ros2.git dwa_planner
cd ~/dwa_ws
colcon build --symlink-install
source install/setup.bash
```

## Run
in order to run the wwhole simulation, you need to source the ros2 install in three different terminals
(source `/opt/ros/humble/setup.bash` and `~/dwa_ws/install/setup.bash`).

1. Gazebo: launches gazebo with the tutrlebot3 burger model
```bash
   export TURTLEBOT3_MODEL=burger
   ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```
2. RViz: use to visualise the markers and the path the robot selects 
   set Fixed Frame to `odom`, add a MarkerArray on `/dwa_markers`
   and a LaserScan on `/scan`.
```bash
   rviz2
```
3. Planner: the dwa planner file consist of the main algorithm, in order to make the robot move from its initial position to the goal use the following command
```bash
   ros2 run dwa_planner dwa_node --ros-args -p goal_x:=-0.8 -p goal_y:=1.1
```
The goal must lie on free floor, at least about 0.4 m from any obstacle.

## Topics
| Topic | Direction | Purpose |
|---|---|---|
| `/odom` | subscribe | robot pose and speed |
| `/scan` | subscribe | laser ranges, converted to obstacle points |
| `/goal_pose` | subscribe | goal clicked in RViz (optional) |
| `/cmd_vel` | publish | chosen velocity command |
| `/dwa_markers` | publish | grey = candidate paths, red = rejected, green = chosen, red sphere = goal |

## Approach
Every 0.1 s the planner:
1. **Samples** an 8 x 15 grid of (linear, angular) velocities inside the
   *dynamic window*: the speeds reachable in one step given acceleration limits.
   
2. **Predicts** a 3s trajectory for each sample (constant-velocity unicycle model).

3. **Rejects** trajectories the robot could not brake to a stop on before the nearest obstacle with the robot radius subtracted.

4. **Scores** the rest with a weighted cost: distance to goal, heading toward the
   goal, obstacle clearance, speed, and smoothness (change of turn rate).
   Heading and clearance terms fade out within about 0.6-0.8 m of the goal so
   the robot can finish its approach.

5. **Publishes** the cheapest velocity on `/cmd_vel` and logs pose, goal distance, nearest obstacle, command and rejected count.

If most paths are blocked and the robot barely moves for 1.5 s, it rotates in
place toward the side with more free space, then resumes. At the goal it stops
and logs `GOAL REACHED`.

## Tuning
Limits, horizon, sample counts, safety distances and cost weights are at the
top of `DwaPlanner.__init__` in `dwa_planner/dwa_node.py`.


