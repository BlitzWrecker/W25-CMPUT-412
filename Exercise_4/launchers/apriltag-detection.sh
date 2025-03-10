#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber
rosrun ex4 apriltag_detection.py

# wait for app to end
dt-launchfile-join