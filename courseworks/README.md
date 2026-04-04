# COMP0250 Coursework 1  
## Pick and Place, Object Detection and Localisation  

---

## 📌 Authors  

- Student A (Zhenggang Chen) Total time consume: 2+5+1
  * Two hours were spent in the lab setting up the environment and running test code, as well as asking questions to the teaching assistants and instructors.
  * Five hours to complete Task 1 (AI statement: The first one hour was spent having ChatGPT5.3 generate the code framework. Two hours were for manual debugging of code and parameters. The remaining two hours were spent communicating with student A and student B in the lab. The entire Task 1 took five hours to complete.)
  * One hour for code integration, testing, checking the rules of instruction, and writing this document.
- Student B (Bingze Li) Total time consume: 2+5
  * Two hours spent communicating with student A about progress and preparing for follow-up tasks.
  * Five hours to complete Task 2 
- Student C (Yihan Wang) Total time consume: 2+5
  * Two hours spent communicating with student A about progress and preparing for follow-up tasks.
  * Five hours to complete Task 3

---

## 📦 Package  

This submission contains the ROS2 package:

cw1_team_14

All solution code is implemented inside this package.

---

## ⚙️ build and run the package 

This project requires ROS2 Humble

```bash
# bash 1
cd ~/comp0250_S26_labs 
source /opt/ros/humble/setup.bash
source install/setup.bash
export PATH=/usr/bin:$PATH 
export RMW_FASTRTPS_USE_SHM=0
colcon build --packages-select cw1_team_14
source /opt/ros/humble/setup.bash
ros2 launch cw1_team_14 run_solution.launch.py  use_gazebo_gui:=true use_rviz:=true  enable_realsense:=true enable_camera_processing:=true  control_mode:=effort
# bash 2
cd ~/comp0250_S26_labs 
source /opt/ros/humble/setup.bash
source install/setup.bash
export PATH=/usr/bin:$PATH 
export RMW_FASTRTPS_USE_SHM=0
ros2 service call /task cw1_world_spawner/srv/TaskSetup "{task_index: 1}"
ros2 service call /task cw1_world_spawner/srv/TaskSetup "{task_index: 2}"
ros2 service call /task cw1_world_spawner/srv/TaskSetup "{task_index: 3}"