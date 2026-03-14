import os

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO

# =========================
# SETTINGS
# =========================

ARUCO_DICT = cv2.aruco.DICT_6X6_50
ARUCO_ID = 0
MARKER_LENGTH_M = 0.10

CONF_THRESHOLD = 0.75

# =========================
# Vision state machine
# =========================

STATE_SEARCH = 0
STATE_WAIT_ROBOT = 1

# =========================
# Robot base offset (camera → robot base)
# manually measured
# =========================

BASE_OFFSET_X = 0.18
BASE_OFFSET_Y = 0.00
BASE_OFFSET_Z = -0.05

# optional test
# BASE_OFFSET_Z = -0.10

# =========================
# Model path
# =========================

package_path = get_package_share_directory("3d_obj_local")
MODEL_PATH = os.path.join(package_path, "best.pt")


class VisionNode(Node):
    def __init__(self):
        super().__init__("vision_node")

        self.pose_pub = self.create_publisher(
            Point,
            "/target_position",
            10
        )

        self.class_pub = self.create_publisher(
            String,
            "/object_class",
            10
        )

        self.get_logger().info("Vision Node Started")

        # =========================
        # Robot done subscriber
        # =========================
        self.robot_done_sub = self.create_subscription(
            Bool,
            "/robot_done",
            self.robot_done_callback,
            10
        )

        # state machine
        self.state = STATE_SEARCH

        # YOLO
        self.model = YOLO(MODEL_PATH)

        # =========================
        # RealSense
        # =========================
        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()

        self.fx = intr.fx
        self.fy = intr.fy
        self.cx = intr.ppx
        self.cy = intr.ppy

        self.K = np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0]
            ],
            dtype=np.float64
        )

        self.dist = np.array(intr.coeffs[:5], dtype=np.float64)

        # =========================
        # ArUco
        # =========================
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        # run every 5 seconds
        self.timer = self.create_timer(5.0, self.vision_loop)

    def vision_loop(self):

        try:

            # =========================
            # Wait if robot is busy
            # =========================
            if self.state == STATE_WAIT_ROBOT:
                return

            frames = self.pipeline.wait_for_frames()
            frames = self.align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                return

            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())

            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = self.aruco_detector.detectMarkers(gray)

            if ids is None:
                return

            ids = ids.flatten()

            rvec = None
            tvec = None

            for i, id_ in enumerate(ids):
                if id_ != ARUCO_ID:
                    continue

                obj_pts = np.array(
                    [
                        [-MARKER_LENGTH_M / 2, MARKER_LENGTH_M / 2, 0.0],
                        [MARKER_LENGTH_M / 2, MARKER_LENGTH_M / 2, 0.0],
                        [MARKER_LENGTH_M / 2, -MARKER_LENGTH_M / 2, 0.0],
                        [-MARKER_LENGTH_M / 2, -MARKER_LENGTH_M / 2, 0.0]
                    ],
                    dtype=np.float32
                )

                img_pts = corners[i][0].astype(np.float32)

                success, rvec, tvec = cv2.solvePnP(
                    obj_pts,
                    img_pts,
                    self.K,
                    self.dist,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                if success:
                    break

            if rvec is None or tvec is None:
                return

            R, _ = cv2.Rodrigues(rvec)
            t = tvec.reshape(3)

            best_obj = None
            best_dist = float("inf")
            best_label = None
            best_conf = None

            results = self.model(color, verbose=False)

            for r in results:

                boxes = r.boxes

                if boxes is None or len(boxes) == 0:
                    continue

                for box in boxes:

                    conf = float(box.conf[0])

                    if conf < CONF_THRESHOLD:
                        continue

                    cls = int(box.cls[0])
                    label = self.model.names[cls]

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    cx_pix = int((x1 + x2) / 2)
                    cy_pix = int((y1 + y2) / 2)

                    if (
                        cx_pix < 0 or cx_pix >= depth.shape[1] or
                        cy_pix < 0 or cy_pix >= depth.shape[0]
                    ):
                        continue

                    Z = float(depth[cy_pix, cx_pix]) * self.depth_scale

                    if Z <= 0.0:
                        continue

                    X = (cx_pix - self.cx) * Z / self.fx
                    Y = (cy_pix - self.cy) * Z / self.fy

                    p_cam = np.array([X, Y, Z], dtype=np.float64)

                    p_marker = R.T @ (p_cam - t)

                    p_robot = np.array([
                        p_marker[0] + BASE_OFFSET_X,
                        p_marker[1] + BASE_OFFSET_Y,
                        p_marker[2] + BASE_OFFSET_Z
                    ])

                    dist = np.linalg.norm(p_robot)

                    if dist < best_dist:
                        best_dist = dist
                        best_obj = p_robot
                        best_label = label
                        best_conf = conf

            if best_obj is not None:

                point_msg = Point()

                point_msg.x = float(best_obj[0])
                point_msg.y = float(best_obj[1])
                point_msg.z = float(best_obj[2])

                self.pose_pub.publish(point_msg)

                class_msg = String()
                class_msg.data = best_label
                self.class_pub.publish(class_msg)

                self.get_logger().info(
                    f"{best_label} Robot base -> X={best_obj[0]:.3f}, "
                    f"Y={best_obj[1]:.3f}, Z={best_obj[2]:.3f}, conf={best_conf:.2f}"
                )

                # =========================
                # robot now executing task
                # =========================
                self.state = STATE_WAIT_ROBOT

        except Exception as e:
            self.get_logger().error(f"Vision loop error: {e}")

    # =========================
    # Robot finished callback
    # =========================
    def robot_done_callback(self, msg):

        if msg.data:
            self.get_logger().info("Robot finished pick and place")

            self.state = STATE_SEARCH

    def destroy_node(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = VisionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
