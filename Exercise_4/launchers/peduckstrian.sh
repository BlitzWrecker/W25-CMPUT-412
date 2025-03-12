#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber
rosrun ex4 peduckstrian.py

# wait for app to end
dt-launchfile-join