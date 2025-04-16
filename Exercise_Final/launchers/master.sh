#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber
rosrun final misc_control.py &
rosrun final master_node.py

# wait for app to end
dt-launchfile-join