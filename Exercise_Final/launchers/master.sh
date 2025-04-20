#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

PARKING_STALL=${PARKING_STALL:-1}

# Debug output
# dts devel run -H csc22928 -L parking -- --env PARKING_STALL=3
echo "Environment PARKING_STALL: ${PARKING_STALL}"
echo "Command line args: $@"

# launch subscriber
rosrun final master_node.py
rosrun final misc_control.py &
rosrun final parking.py "$PARKING_STALL"

# wait for app to end
dt-launchfile-join