"""Object detection node for SO-101 manipulation pipeline.

Subscribes to overhead camera images, detects objects, estimates their
3D position using depth data, and publishes results as PoseStamped messages.

Detection methods:
  - "color": HSV color thresholding (for simulation — fast, reliable)
  - "yolov8": YOLOv8-nano neural network (for real world — Phase 8)

The node publishes the same message types regardless of detection method,
so downstream nodes (pick-and-place) don't care which detector is active.

Published topics:
  /detected_objects  (geometry_msgs/PoseStamped) — 3D position of detected object
  /perception/debug_image  (sensor_msgs/Image) — annotated image for debugging

Usage:
    ros2 run so101_perception detection_node
    ros2 run so101_perception detection_node --ros-args --params-file config/perception.yaml
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image


@dataclass
class Detection2D:
    """A 2D detection result before depth projection."""

    label: str
    confidence: float
    cx: int          # center x in pixels
    cy: int          # center y in pixels
    w: int           # bounding box width
    h: int           # bounding box height


class DetectionNode(Node):
    """ROS2 node for object detection and 3D pose estimation."""

    def __init__(self) -> None:
        super().__init__("detection_node")
        self._bridge = CvBridge()

        # ── Declare parameters with defaults ──────────────────────────
        self.declare_parameter("detection_method", "color")
        self.declare_parameter("color_detection.h_low_1", 0)
        self.declare_parameter("color_detection.h_high_1", 10)
        self.declare_parameter("color_detection.h_low_2", 160)
        self.declare_parameter("color_detection.h_high_2", 180)
        self.declare_parameter("color_detection.s_low", 80)
        self.declare_parameter("color_detection.s_high", 255)
        self.declare_parameter("color_detection.v_low", 80)
        self.declare_parameter("color_detection.v_high", 255)
        self.declare_parameter("color_detection.min_area_pixels", 50)
        self.declare_parameter("camera_frame", "overhead_cam_frame")
        self.declare_parameter("color_topic", "/overhead_cam/color")
        self.declare_parameter("depth_topic", "/overhead_cam/depth")
        self.declare_parameter("camera_info_topic", "/overhead_cam/camera_info")
        self.declare_parameter("detection_topic", "/detected_objects")
        self.declare_parameter("debug_image_topic", "/perception/debug_image")

        # ── Read parameters ───────────────────────────────────────────
        self._method = self.get_parameter("detection_method").value
        self._camera_frame = self.get_parameter("camera_frame").value

        # ── State ─────────────────────────────────────────────────────
        self._camera_info: Optional[CameraInfo] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._depth_header = None

        # ── QoS: use best-effort for high-frequency camera streams ────
        # "Best effort" means if a message is dropped, don't retry.
        # This prevents backlog buildup when processing is slower than
        # the camera frame rate.
        cam_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        # ── Subscribers ───────────────────────────────────────────────
        color_topic = self.get_parameter("color_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        info_topic = self.get_parameter("camera_info_topic").value

        self.create_subscription(CameraInfo, info_topic, self._info_cb, 10)
        self.create_subscription(Image, depth_topic, self._depth_cb, cam_qos)
        self.create_subscription(Image, color_topic, self._color_cb, cam_qos)

        # ── Publishers ────────────────────────────────────────────────
        det_topic = self.get_parameter("detection_topic").value
        debug_topic = self.get_parameter("debug_image_topic").value

        self._det_pub = self.create_publisher(PoseStamped, det_topic, 10)
        self._debug_pub = self.create_publisher(Image, debug_topic, 1)

        self.get_logger().info(
            f"Detection node started — method={self._method}, "
            f"camera_frame={self._camera_frame}"
        )

    # ── Callbacks ─────────────────────────────────────────────────────

    def _info_cb(self, msg: CameraInfo) -> None:
        """Store camera intrinsics (only need one message)."""
        if self._camera_info is None:
            self._camera_info = msg
            self.get_logger().info(
                f"Camera info received: {msg.width}x{msg.height}, "
                f"fx={msg.k[0]:.1f}, fy={msg.k[4]:.1f}"
            )

    def _depth_cb(self, msg: Image) -> None:
        """Store latest depth frame for use when color frame arrives."""
        self._latest_depth = np.frombuffer(
            msg.data, dtype=np.float32,
        ).reshape(msg.height, msg.width)
        self._depth_header = msg.header

    def _color_cb(self, msg: Image) -> None:
        """Process color frame: detect objects and publish 3D poses."""
        if self._camera_info is None:
            return  # Not ready — waiting for camera info

        # Convert ROS Image → OpenCV BGR
        color_rgb = np.frombuffer(
            msg.data, dtype=np.uint8,
        ).reshape(msg.height, msg.width, 3)
        color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)

        # ── Run detection ─────────────────────────────────────────────
        if self._method == "color":
            detections = self._detect_color(color_bgr)
        else:
            self.get_logger().warn(f"Unknown detection method: {self._method}")
            return

        # ── Project to 3D and publish ─────────────────────────────────
        debug_frame = color_bgr.copy()

        for det in detections:
            # Draw bounding box on debug image
            x1 = det.cx - det.w // 2
            y1 = det.cy - det.h // 2
            x2 = det.cx + det.w // 2
            y2 = det.cy + det.h // 2
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                debug_frame, f"{det.label} {det.confidence:.2f}",
                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

            # Project to 3D using depth
            pose_3d = self._project_to_3d(det, msg.header)
            if pose_3d is not None:
                self._det_pub.publish(pose_3d)
                p = pose_3d.pose.position
                cv2.putText(
                    debug_frame,
                    f"({p.x:.3f}, {p.y:.3f}, {p.z:.3f})",
                    (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1,
                )

        # Publish debug image
        debug_msg = self._bridge.cv2_to_imgmsg(debug_frame, encoding="bgr8")
        debug_msg.header = msg.header
        self._debug_pub.publish(debug_msg)

    # ── Detection methods ─────────────────────────────────────────────

    def _detect_color(self, bgr: np.ndarray) -> list[Detection2D]:
        """Detect objects by HSV color thresholding.

        Works reliably in simulation where colors are consistent.
        For real-world, swap to _detect_yolov8 (Phase 8).

        Pipeline:
          1. Convert BGR → HSV color space
          2. Threshold for target color (red has two HSV ranges)
          3. Find contours in the binary mask
          4. Filter by minimum area
          5. Return bounding box center + size
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Read HSV thresholds from parameters
        h_lo1 = self.get_parameter("color_detection.h_low_1").value
        h_hi1 = self.get_parameter("color_detection.h_high_1").value
        h_lo2 = self.get_parameter("color_detection.h_low_2").value
        h_hi2 = self.get_parameter("color_detection.h_high_2").value
        s_lo = self.get_parameter("color_detection.s_low").value
        s_hi = self.get_parameter("color_detection.s_high").value
        v_lo = self.get_parameter("color_detection.v_low").value
        v_hi = self.get_parameter("color_detection.v_high").value
        min_area = self.get_parameter("color_detection.min_area_pixels").value

        # Red wraps around 0° in HSV, so we need two ranges
        mask1 = cv2.inRange(hsv, (h_lo1, s_lo, v_lo), (h_hi1, s_hi, v_hi))
        mask2 = cv2.inRange(hsv, (h_lo2, s_lo, v_lo), (h_hi2, s_hi, v_hi))
        mask = mask1 | mask2

        # Morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[Detection2D] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            detections.append(Detection2D(
                label="cube",
                confidence=min(1.0, area / 500.0),  # Rough confidence from area
                cx=x + w // 2,
                cy=y + h // 2,
                w=w,
                h=h,
            ))

        return detections

    # ── 3D Projection ─────────────────────────────────────────────────

    def _project_to_3d(
        self, det: Detection2D, color_header,
    ) -> Optional[PoseStamped]:
        """Project a 2D detection to 3D using depth image + camera intrinsics.

        Math:
          Given pixel (u, v) and depth d, the 3D point in camera frame is:
            X = (u - cx) * d / fx
            Y = (v - cy) * d / fy
            Z = d
          Where fx, fy are focal lengths and (cx, cy) is the principal point,
          all from the camera intrinsics matrix K.

        The result is in the camera frame (overhead_cam_frame). Downstream
        nodes can use TF2 to transform to base_link if needed.
        """
        if self._latest_depth is None:
            self.get_logger().warn("No depth data available", throttle_duration_sec=5.0)
            return None

        if self._camera_info is None:
            return None

        depth = self._latest_depth
        h, w = depth.shape

        # Clamp detection center to image bounds
        u = max(0, min(det.cx, w - 1))
        v = max(0, min(det.cy, h - 1))

        # Sample depth in a small region around the center (more robust than single pixel)
        half_k = 3  # 7x7 patch
        v_lo = max(0, v - half_k)
        v_hi = min(h, v + half_k + 1)
        u_lo = max(0, u - half_k)
        u_hi = min(w, u + half_k + 1)
        patch = depth[v_lo:v_hi, u_lo:u_hi]

        # Filter out invalid depths (0 or very large)
        valid = patch[(patch > 0.01) & (patch < 10.0)]
        if len(valid) == 0:
            self.get_logger().warn("No valid depth at detection center", throttle_duration_sec=5.0)
            return None

        d = float(np.median(valid))

        # Camera intrinsics from K matrix
        # K = [fx  0  cx]
        #     [0  fy  cy]
        #     [0   0   1]
        k = self._camera_info.k
        fx, fy = k[0], k[4]
        cx_cam, cy_cam = k[2], k[5]

        if fx == 0 or fy == 0:
            self.get_logger().error("Camera intrinsics have zero focal length")
            return None

        # Back-project to 3D (camera frame)
        x_3d = (u - cx_cam) * d / fx
        y_3d = (v - cy_cam) * d / fy
        z_3d = d

        # Build PoseStamped message
        pose = PoseStamped()
        pose.header.stamp = color_header.stamp
        pose.header.frame_id = self._camera_frame
        pose.pose.position.x = float(x_3d)
        pose.pose.position.y = float(y_3d)
        pose.pose.position.z = float(z_3d)
        # Orientation: identity (we don't estimate object orientation)
        pose.pose.orientation.w = 1.0

        return pose


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
