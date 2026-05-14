import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Header
import pywt
import numpy as np

class ImpactVerifierNode(Node):
    def __init__(self):
        super().__init__('impact_verifier_node')

        # Declare parameters
        self.declare_parameter('wheelbase', 0.14)
        self.declare_parameter('track_width_front', 0.14)
        self.declare_parameter('track_width_rear', 0.145)
        self.declare_parameter('wheel_width', 0.025)
        self.declare_parameter('wheel_radius', 0.0325)
        self.declare_parameter('window_size', 64)
        self.declare_parameter('impact_threshold_fixed', 2.5)
        self.declare_parameter('impact_threshold_dynamic_n_sigma', 3.0)

        # Get parameters
        self.wheelbase = self.get_parameter('wheelbase').value
        self.track_width_front = self.get_parameter('track_width_front').value
        self.track_width_rear = self.get_parameter('track_width_rear').value
        self.wheel_width = self.get_parameter('wheel_width').value
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.window_size = self.get_parameter('window_size').value
        self.impact_threshold_fixed = self.get_parameter('impact_threshold_fixed').value
        self.impact_threshold_dynamic_n_sigma = self.get_parameter('impact_threshold_dynamic_n_sigma').value

        self.get_logger().info(f"Initialized ImpactVerifierNode with window_size={self.window_size}, "
                               f"impact_threshold_fixed={self.impact_threshold_fixed}, "
                               f"wheelbase={self.wheelbase}")

        # Buffer for Z-axis acceleration
        self.z_accel_buffer = []

        # Temporal cooldown variables
        self.last_impact_time = None
        self.cooldown_duration_sec = 0.2

        # Publisher for hazard trigger
        self.hazard_pub = self.create_publisher(Header, '/aida/hazard_trigger', 10)

        # Subscriber to IMU data
        # Using a queue size of 100 since data is 100Hz
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            100
        )

    def imu_callback(self, msg: Imu):
        z_accel = msg.linear_acceleration.z
        self.z_accel_buffer.append(z_accel)

        # Process when buffer is full
        if len(self.z_accel_buffer) >= self.window_size:
            self.process_buffer(msg.header.stamp)

            # Slide window by 1 or clear buffer?
            # A sliding window of 1 might be computationally expensive for PyWavelets DWT at 100Hz,
            # but let's slide by half the window size or just clear the oldest element.
            # To be safe and ensure rolling window as requested, we pop the oldest.
            if len(self.z_accel_buffer) > 0:
                self.z_accel_buffer.pop(0)

    def process_buffer(self, stamp):
        signal = np.array(self.z_accel_buffer)

        # Perform Discrete Wavelet Transform
        # Wavelet: db4
        # We can do a single level decomposition or multi-level. We'll do level 1 for now.
        coeffs = pywt.wavedec(signal, 'db4', level=1)

        # In pywt.wavedec, the last element is the level 1 detail coefficients (high frequency)
        detail_coeffs = coeffs[-1]

        # Thresholding logic
        # 1. Fixed threshold
        max_magnitude = np.max(np.abs(detail_coeffs))

        # 2. Dynamic threshold (Stretch Goal)
        mean = np.mean(detail_coeffs)
        std_dev = np.std(detail_coeffs)
        dynamic_threshold = mean + (self.impact_threshold_dynamic_n_sigma * std_dev)

        # We will use the fixed threshold as requested, but also compute the dynamic one.
        # Let's say if it passes the fixed threshold, it's a hazard.
        is_hazard = max_magnitude > self.impact_threshold_fixed

        if is_hazard:
            # Check temporal cooldown
            current_time_sec = stamp.sec + (stamp.nanosec * 1e-9)

            if self.last_impact_time is None or (current_time_sec - self.last_impact_time) >= self.cooldown_duration_sec:
                self.get_logger().info(
                    f"Impact verified! Max mag: {max_magnitude:.2f} > Threshold: {self.impact_threshold_fixed:.2f}"
                )

                hazard_msg = Header()
                hazard_msg.stamp = stamp
                hazard_msg.frame_id = 'imu_link' # Assuming standard frame_id or we could extract from imu msg
                self.hazard_pub.publish(hazard_msg)

                self.last_impact_time = current_time_sec

def main(args=None):
    rclpy.init(args=args)
    node = ImpactVerifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
