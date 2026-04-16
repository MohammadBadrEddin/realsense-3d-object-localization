# RealSense 3D Object Localization (ROS2)

Real-time vision system for robotic pick-and-place using Intel RealSense D435i, YOLO, and ROS2.

---

## Purpose

Detect objects and publish their 3D position in robot base frame for grasping.

---

## Pipeline

RGB-D → YOLO → Depth → 3D (camera) → ArUco (ID=0, 6x6_50) → Robot base → ROS2

---

## Features
2D → 3D localization (depth + intrinsics)
ArUco-based frame transformation
Closest-object selection + forbidden zone filtering
State control (wait while robot executes)
Model
YOLO (best.pt), confidence ≥ 0.75
Replaceable with any custom-trained model
ROS2 Topics

---

## Publishes

/target_position (geometry_msgs/Point)
/object_class (std_msgs/String)
/camera_done (std_msgs/Bool)

---

## Subscribes

/task_complete (std_msgs/Bool)

---

## Output

3D position + class label for grasping

---


## Example:
cube → X=0.51, Y=0.13, Z=0.09, conf=0.91