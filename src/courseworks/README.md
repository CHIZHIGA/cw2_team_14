# COMP0250 Coursework 1  

Github repo: https://github.com/CHIZHIGA/cw2_team_14

This coursework has been run on both WSL and virtual machines. 
The success rate of the Task1 exceeded 90%.  
The success rate of the Task2 is 100%.  
The success rate of the Task3 exceeded 90%.  
Therefore, in such cases, we hope that the instructor and teaching assistants can run the code multiple times to observe the results and structure.

## Pick and Place, Object Detection and Localisation  

---

## 📌 Authors  

- Student A (Zhenggang Chen) Total time consume: 2+5+5+3
  * 2 hours for Initial commit for Task1 on Apr 4, 2026;
  * 5 hours for Task2 on Apr 9, 2026;
  * 5 hours for Task3 recognize problem on Apr 17, 2026;
  * 3 hours for Task3 on Apr 25, 2026.
- Student B (Yihan Wang) Total time consume: 5+10
  * 5 hours for Task3 on Apr 9, 2026;
  * 10 hours for Task3 pick and place problem on Apr 24, 2026;
- Student C (Bingze Li) Total time consume: 6+11
  * 6 hours for Task1 demo on Apr 8, 2026;
  * 11 hours for Task1 on Apr 20, 2026.

---

## 📦 Package  

This submission contains the ROS2 package:

cw2_team_14

All solution code is implemented inside this package.

---

## ⚙️ build and run the package 

This project requires ROS2 Humble

```bash
# bash 1
source install/setup.bash
colcon build --packages-select cw1_team_14
ros2 launch cw2_team_14 run_solution.launch.py   use_gazebo_gui:=true use_rviz:=false
# bash 2
source install/setup.bash
ros2 service call /task cw2_world_spawner/srv/TaskSetup "{task_index: 1}"
ros2 service call /task cw2_world_spawner/srv/TaskSetup "{task_index: 2}"
ros2 service call /task cw2_world_spawner/srv/TaskSetup "{task_index: 3}"
