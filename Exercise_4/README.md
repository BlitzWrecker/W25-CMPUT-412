# Exercise 3: Computer Vision and Autonomous Navigation

This repository is built on the following template: [Duckietown Template-ROS](https://github.com/duckietown/template-ros/).

In this exercise, we explore computer vision techniques for lane detection, color-based behavior execution, and autonomous navigation using controllers. The project is divided into three main parts: **Computer Vision**, **Controllers**, and **Lane Following**.

---

## Part 1: Computer Vision👀

### Camera Distortion Correction and Color Detection
**File:** `packages/computer_vision/src/lane_detection.py`  
**Description:** Corrects camera distortion using intrinsic parameters, detects blue, red, and green lanes using HSV color thresholds, performs contouring. Publishes to processed_image.

### Lane Behavior
**File:** `packages/computer_vision/src/lane_based_behavior_controller.py`  
**Description:** executes lane-specific behaviors (e.g., stopping, turning, signaling LEDs).

▶ **Launch Command:**
```bash
dts devel run -H ROBOT_NAME -L lane-based-behavior
```

---

## Part 2: Controllers🕹️


**File:** `packages/computer_vision/src/lane_following_controller.py`  
**Description:** Implements P, PD, and PID controllers for lane following along a straight path for 1.5 meters.

▶ **Launch Command:**
```bash
dts devel run -H ROBOT_NAME -L lane-follow-controller
```

---

## Part 3: Lane Following🏁

### Lane Following Node
**File:** `packages/computer_vision/src/lane_following.py`  
**Description:** Integrates computer vision and controllers to perform full-lap lane following using OpenCV.

▶ **Launch Command:**
```bash
dts devel run -H ROBOT_NAME -L lane-following
```

---

## Bonus: English Driving☕️

**File:** `packages/computer_vision/src/english_driving.py`  
**Description:** This is almost identical to lane_following.py, except the duckiebot drives on the left side of the road instead. It also performs full-lap lane following.

▶ **Launch Command:**
```bash
dts devel run -H ROBOT_NAME -L english-driving
```

---
