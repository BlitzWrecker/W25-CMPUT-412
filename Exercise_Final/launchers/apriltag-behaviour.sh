#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber
rosrun ex4 misc_control.py &
rosrun ex4 apriltag_detection.py &
rosrun ex4 apriltag_behaviour.py

# wait for app to end
dt-launchfile-join