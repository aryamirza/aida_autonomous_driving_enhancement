import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

import time
import math

class AidaNavNode(Node):
    def __init__(self):
        super().__init__('aida_nav_node')

        # --- Parameters ---
        # PID Constants
        self.declare_parameter('kp', 0.5)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.1)
        self.declare_parameter('gimbal_kp', 0.20)

        # ROI Proportions
        self.declare_parameter('roi_top_width', 0.4)
        self.declare_parameter('roi_bottom_width', 0.9)
        self.declare_parameter('roi_height_start', 0.6)

        self.declare_parameter('v_max', 0.22)
        self.declare_parameter('v_min', 0.12)

        self.kp = self.get_parameter('kp').value
        self.ki = self.get_parameter('ki').value
        self.kd = self.get_parameter('kd').value
        self.gimbal_kp = self.get_parameter('gimbal_kp').value

        self.roi_top_width = self.get_parameter('roi_top_width').value
        self.roi_bottom_width = self.get_parameter('roi_bottom_width').value
        self.roi_height_start = self.get_parameter('roi_height_start').value

        # --- State Variables ---
        self.bridge = CvBridge()
        self.last_valid_time = self.get_clock().now()
        self.line_found = False
        self.last_cte = 0.0
        self.normalized_cte = 0.0
        self.current_yaw_rate = 0.0  # Used for adaptive lookahead and speed

        # PID state
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.last_pid_time = self.get_clock().now()

        # Latest Odometry speed (if needed, though we set it directly)
        self.current_speed = 0.0

        # --- Subscribers & Publishers ---
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel_nav', 10)
        self.gimbal_pub = self.create_publisher(JointState, '/camera/gimbal_cmd', 10)

        # --- Timer (30Hz) ---
        self.timer = self.create_timer(1.0 / 30.0, self.control_loop)

    def odom_callback(self, msg):
        self.current_speed = msg.twist.twist.linear.x


    def get_warp_matrices(self, width, height):
        # Calculate trapezoid points
        top_y = int(height * self.roi_height_start)
        bottom_y = height

        top_w = int(width * self.roi_top_width)
        bottom_w = int(width * self.roi_bottom_width)

        src_pts = np.float32([
            [(width - top_w) // 2, top_y],
            [(width + top_w) // 2, top_y],
            [(width - bottom_w) // 2, bottom_y],
            [(width + bottom_w) // 2, bottom_y]
        ])

        dst_pts = np.float32([
            [0, 0],
            [width, 0],
            [0, height],
            [width, height]
        ])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
        return M, M_inv

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return

        height, width = cv_image.shape

        # Binary Thresholding (Otsu's)
        _, binary = cv2.threshold(cv_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Inverse Perspective Mapping
        M, _ = self.get_warp_matrices(width, height)
        bev_image = cv2.warpPerspective(binary, M, (width, height), flags=cv2.INTER_LINEAR)

        # Adaptive Look-Ahead
        # Straight: search down to y=40% (0.4)
        # Full turn (|w_z| >= 1.0): search down to y=70% (0.7)
        yaw_rate = min(abs(self.current_yaw_rate), 1.0)
        search_min_y_ratio = 0.4 + (0.7 - 0.4) * yaw_rate
        search_min_y = int(height * search_min_y_ratio)

        # Sliding Window Setup
        histogram = np.sum(bev_image[height//2:, :], axis=0)
        midpoint = int(histogram.shape[0] // 2)

        # We will try to find a single line (lane center) or assume it's the strongest signal
        base_x = np.argmax(histogram)

        if histogram[base_x] == 0:
            self.line_found = False
            return

        nwindows = 9
        window_height = int((height - search_min_y) / nwindows)
        margin = 50
        minpix = 20

        nonzero = bev_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        current_x = base_x
        lane_inds = []

        # Slide windows from bottom up to search_min_y
        for window in range(nwindows):
            win_y_low = height - (window + 1) * window_height
            win_y_high = height - window * window_height

            win_x_low = current_x - margin
            win_x_high = current_x + margin

            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

            lane_inds.append(good_inds)

            if len(good_inds) > minpix:
                current_x = int(np.mean(nonzerox[good_inds]))

        lane_inds = np.concatenate(lane_inds)

        if len(lane_inds) == 0:
            self.line_found = False
            return

        x_points = nonzerox[lane_inds]
        y_points = nonzeroy[lane_inds]

        try:
            poly_fit = np.polyfit(y_points, x_points, 2)
        except Exception as e:
            self.line_found = False
            return

        self.line_found = True
        self.last_valid_time = self.get_clock().now()

        # Adaptive CTE calculation
        # Target point Y: 50% for straight, 80% for max turn
        target_y_ratio = 0.5 + (0.8 - 0.5) * yaw_rate
        target_y = int(height * target_y_ratio)

        # Calculate polynomial value at target_y
        target_x = poly_fit[0]*target_y**2 + poly_fit[1]*target_y + poly_fit[2]

        # CTE is error relative to the center of the image at the target Y
        center_x = width / 2.0
        self.last_cte = center_x - target_x

        # Normalized CTE [-1.0, 1.0] based on half ROI width (using image width / 2 as approximation for max error)
        # Assuming maximum pixel error is half the image width
        self.normalized_cte = np.clip(self.last_cte / (width / 2.0), -1.0, 1.0)



    def control_loop(self):
        # Fetch parameters dynamically for real-time tuning
        self.kp = self.get_parameter('kp').value
        self.ki = self.get_parameter('ki').value
        self.kd = self.get_parameter('kd').value
        self.gimbal_kp = self.get_parameter('gimbal_kp').value

        self.roi_top_width = self.get_parameter('roi_top_width').value
        self.roi_bottom_width = self.get_parameter('roi_bottom_width').value
        self.roi_height_start = self.get_parameter('roi_height_start').value

        now = self.get_clock().now()
        dt_duration = now - self.last_pid_time
        dt = dt_duration.nanoseconds / 1e9
        self.last_pid_time = now

        if dt <= 0.0:
            dt = 1.0 / 30.0

        time_since_valid = (now - self.last_valid_time).nanoseconds / 1e9

        cmd_vel = Twist()
        gimbal_cmd = JointState()
        gimbal_cmd.name = ["pan", "tilt"]

        # 1. Fail-Safe & Recovery
        if not self.line_found or time_since_valid > 0.3:
            # Coast for 0.3s or halt
            if time_since_valid <= 0.3:
                # Maintain last steering, drop speed to 0.10
                cmd_vel.angular.z = self.current_yaw_rate
                cmd_vel.linear.x = 0.10
            else:
                # Halt
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.0
                self.current_yaw_rate = 0.0

            self.cmd_vel_pub.publish(cmd_vel)
            # Center gimbal as fallback
            gimbal_cmd.position = [0.0, 0.0]
            self.gimbal_pub.publish(gimbal_cmd)
            return

        # 2. Steering (PID)
        error = self.last_cte

        self.integral_error += error * dt
        derivative = (error - self.prev_error) / dt

        angular_z = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * derivative)

        # Invert logic depending on CTE definition (usually positive CTE to left means turn right, need to test)
        # Assuming last_cte = center_x - target_x. If target is to the right (target_x > center_x), CTE is negative.
        # If target is left, CTE is positive. Positive angular.z turns left in ROS.
        # So positive CTE -> target is left -> turn left -> positive angular.z.
        cmd_vel.angular.z = angular_z
        self.current_yaw_rate = angular_z
        self.prev_error = error

        # 3. Propulsion (Dynamic Speed)
        yaw_rate_clamped = min(abs(angular_z), 1.0)
        v_max = self.get_parameter('v_max').value
        v_min = self.get_parameter('v_min').value
        cmd_vel.linear.x = v_max - (v_max - v_min) * (yaw_rate_clamped / 1.0)

        # 4. Gimbal Gaze (Pan/Tilt)
        # Tilt: linearly map from v_min (-10 deg) to v_max (+15 deg)
        # tilt_deg = tilt_min + (tilt_max - tilt_min) * (vx - v_min) / (v_max - v_min)
        tilt_deg = -10.0 + (15.0 - -10.0) * (cmd_vel.linear.x - v_min) / (v_max - v_min)
        tilt_rad = math.radians(tilt_deg)

        # Pan: Gimbal Kp * Maximum Pan Offset (e.g., 30 deg) * normalized_cte
        # Pan left (positive angle) when CTE is positive (line is left)
        # We use a theoretical max pan of 30 degrees scaled by gimbal_kp
        pan_deg = 30.0 * self.gimbal_kp * self.normalized_cte
        pan_rad = math.radians(pan_deg)

        gimbal_cmd.position = [pan_rad, tilt_rad]

        # Publish
        self.cmd_vel_pub.publish(cmd_vel)
        self.gimbal_pub.publish(gimbal_cmd)


def main(args=None):
    rclpy.init(args=args)
    node = AidaNavNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
