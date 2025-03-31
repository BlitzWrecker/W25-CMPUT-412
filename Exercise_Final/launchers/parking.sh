#!/bin/bash


source /environment.sh


dt-launchfile-init


# Read from both possible sources (env var and command line)
PARKING_STALL=${PARKING_STALL:-1}  # Default to 1 if not set


# Debug output
# dts devel run -H csc22928 -L parking -- --env PARKING_STALL=3
echo "Environment PARKING_STALL: ${PARKING_STALL}"
echo "Command line args: $@"


rosrun ex4 parking.py "$PARKING_STALL"


dt-launchfile-join