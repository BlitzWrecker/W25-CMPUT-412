cmake_minimum_required(VERSION 2.8.3)
project(ex4)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  message_generation
)

## Declare the service files to be processed
add_service_files(
  FILES
  MiscCtrlCMD.srv
)

add_message_files(
  FILES
  NavigateCMD.msg
)

## Generate messages
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

## Declare a catkin package
catkin_package(
  CATKIN_DEPENDS message_runtime rospy std_msgs
)
