#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# Read parking stall number from environment variable
PARKING_STALL=${PARKING_STALL:-1}  # Default to stall 1 if not specified

# Pass the parking stall number to the Python script
rosrun ex4 parking.py ${PARKING_STALL}

# wait for app to end
dt-launchfile-join