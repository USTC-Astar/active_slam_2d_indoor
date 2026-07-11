cmake_minimum_required(VERSION 3.0.2)
project(indoor_cartographer_demo)

find_package(catkin REQUIRED)
catkin_package()

catkin_install_python(PROGRAMS
  scripts/active_slam_explorer.py
  scripts/indoor_cmd_vel_safety.py
  scripts/indoor_odom_filter.py
  scripts/indoor_spawn_or_replace_model.py
  scripts/indoor_waypoint_mapper.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
