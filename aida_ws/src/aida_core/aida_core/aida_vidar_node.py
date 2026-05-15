import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, JointState
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String
import message_filters
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import tf2_ros
import threading
import sys
import termios
import tty
from datetime import datetime, timezone
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Twist

class AidaVidarNode(Node):
    def __init__(self):
        super().__init__('aida_vidar_node')

        self.is_engaged = False

        # Declare Parameters
        self.declare_parameter('camera_fov', 1.39)
        self.declare_parameter('camera_pivot_z', 0.085)
        self.declare_parameter('camera_pivot_offset_x', -0.01)
        self.declare_parameter('lidar_offset_x', -0.085)

        self.camera_fov = self.get_parameter('camera_fov').value
        self.camera_pivot_offset_x = self.get_parameter('camera_pivot_offset_x').value
        self.lidar_offset_x = self.get_parameter('lidar_offset_x').value

        # Setup Publishers & Subscribers
        self.snapshot_pub = self.create_publisher(String, '/aida/vidar/snapshots', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()

        # TF2 Setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Setup Synchronizer
        yolo_sub = message_filters.Subscriber(self, Detection2DArray, '/yolov5_ros2/object_detect')
        scan_sub = message_filters.Subscriber(self, LaserScan, '/scan_raw')
        odom_sub = message_filters.Subscriber(self, Odometry, '/odom')
        joint_sub = message_filters.Subscriber(self, JointState, '/joint_states')
        img_sub = message_filters.Subscriber(self, Image, '/usb_cam/image_raw')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [yolo_sub, scan_sub, odom_sub, joint_sub, img_sub],
            queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)

        # Start keyboard listener thread
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_thread.start()

    def keyboard_listener(self):
        # Save terminal settings
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while True:
                char = sys.stdin.read(1).lower()
                if char == 's' and not self.is_engaged:
                    self.is_engaged = True
                    self.get_logger().info('[INFO] AUTONOMY ENGAGED - IGNITION START')
                elif char == 'a' and self.is_engaged:
                    self.is_engaged = False
                    self.get_logger().warn('[WARN] ABORT TRIGGERED - VEHICLE HALTED')
                    self.publish_zero_velocity()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def publish_zero_velocity(self):
        zero_twist = Twist()
        zero_twist.linear.x = 0.0
        zero_twist.angular.z = 0.0
        self.cmd_vel_pub.publish(zero_twist)

    def get_pan_tilt(self, joint_msg):
        pan = 0.0
        tilt = 0.0
        for i, name in enumerate(joint_msg.name):
            if name == 'pan_joint':
                pan = joint_msg.position[i]
            elif name == 'tilt_joint':
                tilt = joint_msg.position[i]
        return pan, tilt

    def get_camera_transform(self, time):
        try:
            trans = self.tf_buffer.lookup_transform(
                'base_link', 'camera_optical_frame', time, rclpy.duration.Duration(seconds=0.1)
            )
            return trans
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"TF2 Lookup failed: {e}")
            return None

    def quaternion_to_pitch(self, q):
        # Convert quaternion to pitch (rotation around Y axis)
        sinp = 2 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1:
            pitch = np.sign(sinp) * (np.pi / 2) # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
        return pitch

    def sync_callback(self, yolo_msg, scan_msg, odom_msg, joint_msg, img_msg):
        # Idle State: Brakes locked if not engaged
        if not self.is_engaged:
            self.publish_zero_velocity()
            return

        # 1. TF2 Transform Lookup
        # Convert builtin_interfaces.msg.Time to rclpy.time.Time
        time_now = rclpy.time.Time.from_msg(img_msg.header.stamp)
        transform = self.get_camera_transform(time_now)
        if not transform:
            return

        # Extract pitch and height from TF2
        q = transform.transform.rotation

        # For optical frame (Z forward, X right, Y down):
        # We transform the Z-forward unit vector [0, 0, 1] by the quaternion to get its direction in base_link.
        # This will give us the actual pitch.
        x = q.x; y = q.y; z = q.z; w = q.w
        # The z-component of the transformed [0, 0, 1] vector:
        vz = 2.0 * (x*z - w*y)
        cam_pitch = np.arcsin(-vz) if vz < 1.0 else np.arcsin(-1.0)

        cam_height = transform.transform.translation.z

        pan, tilt = self.get_pan_tilt(joint_msg)

        # We'll maintain a list of detected anomalies to avoid overlapping detections
        detected_anomalies = []

        # Convert Image
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        height, width, _ = cv_image.shape

        # Calculate vertical FOV based on aspect ratio
        vertical_fov = self.camera_fov * (height / width)

        # 2. Semantic-Geometric Fusion (YOLO Detections)
        for detection in yolo_msg.detections:
            center_x = detection.bbox.center.x
            center_y = detection.bbox.center.y
            label = detection.results[0].hypothesis.class_id if detection.results else "unknown"
            score = detection.results[0].hypothesis.score if detection.results else 0.0

            # Map pixel to bearing (negative because right pixel means negative angle in ROS/LiDAR)
            theta = -((center_x - (width / 2.0)) / width) * self.camera_fov
            # Adjust for pan joint
            adjusted_theta = theta + pan

            # Find closest LiDAR distance at this angle
            angle_min = scan_msg.angle_min
            angle_increment = scan_msg.angle_increment

            # Ensure adjusted_theta is within LiDAR range
            if scan_msg.angle_min <= adjusted_theta <= scan_msg.angle_max:
                index = int((adjusted_theta - angle_min) / angle_increment)
                if 0 <= index < len(scan_msg.ranges):
                    dist = scan_msg.ranges[index]

                    if not np.isinf(dist) and not np.isnan(dist):
                        # NumPy vectorized coordinate transformation
                        # [local_x, local_y] = dist * [cos(theta), sin(theta)]
                        local_coords = dist * np.array([np.cos(adjusted_theta), np.sin(adjusted_theta)])
                        local_x, local_y = local_coords[0] + self.lidar_offset_x, local_coords[1]

                        # Apply to Global Coordinates
                        self.publish_anomaly(label, score, local_x, local_y, odom_msg, pan, tilt)

                        # Add to our list to prevent OpenCV fallback taking this area
                        detected_anomalies.append((center_x, center_y))

        # 3. OpenCV Fallback for Cracks (Bottom 30%)
        # Only process if we don't have YOLO detections in the impact zone
        impact_zone_start = int(height * 0.7)
        yolo_in_zone = any(y > impact_zone_start for x, y in detected_anomalies)

        if not yolo_in_zone:
            roi = cv_image[impact_zone_start:height, 0:width]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

            if lines is not None:
                # Find the most prominent line
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # We use the midpoint of the line
                    mid_x = (x1 + x2) / 2.0
                    mid_y = (y1 + y2) / 2.0 + impact_zone_start

                    # Inverse Perspective Mapping using TF2 Pitch and Height
                    # Simple pinhole camera approximation
                    theta_y = ((mid_y - (height / 2.0)) / height) * vertical_fov
                    total_pitch = cam_pitch + theta_y

                    if total_pitch > 0: # Looking down
                        # NumPy trig operations
                        dist_x_local = cam_height / np.tan(total_pitch)

                        # Negative for right-side pixels to match ROS coordinates
                        theta_x = -((mid_x - (width / 2.0)) / width) * self.camera_fov
                        dist_y_local = dist_x_local * np.tan(theta_x + pan)

                        # Adjust for camera pivot offset
                        dist_x_local += self.camera_pivot_offset_x

                        self.publish_anomaly("crack", 0.5, dist_x_local, dist_y_local, odom_msg, pan, tilt)
                        break # Only process one prominent crack per frame

    def publish_anomaly(self, label, confidence, local_x, local_y, odom_msg, pan, tilt):
        # Transform local to global using NumPy
        robot_x = odom_msg.pose.pose.position.x
        robot_y = odom_msg.pose.pose.position.y

        # Quaternion to yaw using NumPy
        q = odom_msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        # Rotation matrix for local to global
        rot_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw),  np.cos(yaw)]
        ])

        local_vec = np.array([local_x, local_y])
        global_vec = np.array([robot_x, robot_y]) + np.dot(rot_matrix, local_vec)

        global_x, global_y = global_vec[0], global_vec[1]

        payload = {
            "label": label,
            "confidence_init": float(confidence),
            "global_x": float(global_x),
            "global_y": float(global_y),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "pan": float(pan),
                "tilt": float(tilt)
            }
        }

        msg = String()
        msg.data = json.dumps(payload)
        self.snapshot_pub.publish(msg)
        self.get_logger().info(f"Published Anomaly: {label} at ({global_x:.2f}, {global_y:.2f})")

def main(args=None):
    rclpy.init(args=args)
    node = AidaVidarNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
