import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import json
import math
import time

STATE_NORMAL = 0
STATE_BRACE = 1

class AidaBraceNode(Node):
    def __init__(self):
        super().__init__('aida_brace_node')

        # Parameters
        self.declare_parameter('max_steering_limit', 0.5)

        # Publishers and Subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.nav_sub = self.create_subscription(Twist, '/cmd_vel_nav', self.nav_callback, 10)
        self.vidar_sub = self.create_subscription(Twist, '/cmd_vel_vidar', self.vidar_callback, 10)
        self.hazard_sub = self.create_subscription(String, '/aida/hazard_warning', self.hazard_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # State variables
        self.state = STATE_NORMAL
        self.last_nav_msg = None
        self.last_nav_time = time.time()

        self.last_vidar_msg = None
        self.last_vidar_time = 0.0

        # Brace state variables
        self.brace_protocol = None
        self.brace_start_x = 0.0
        self.brace_start_y = 0.0
        self.current_x = 0.0
        self.current_y = 0.0
        self.steering_offset = 0.0

        # Distance targets for each protocol
        self.protocol_distances = {
            "speedbump": 0.40,
            "small_bump": 0.25,
            "crack": 0.20
        }

        # Watchdog timer
        self.create_timer(0.05, self.watchdog_timer_callback)

    @property
    def max_steering_limit(self):
        return self.get_parameter('max_steering_limit').get_parameter_value().double_value

    def clamp_steering(self, value):
        limit = self.max_steering_limit
        return max(-limit, min(limit, value))

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        if self.state == STATE_BRACE and self.brace_protocol in self.protocol_distances:
            dist = math.sqrt((self.current_x - self.brace_start_x)**2 + (self.current_y - self.brace_start_y)**2)
            target_dist = self.protocol_distances[self.brace_protocol]

            if dist >= target_dist:
                self.get_logger().info(f'{self.brace_protocol.capitalize()} cleared (distance {dist:.2f} >= {target_dist}), returning to normal.')
                self.state = STATE_NORMAL
                self.brace_protocol = None
                self.publish_cmd()

    def hazard_callback(self, msg):
        try:
            data = json.loads(msg.data)
            label = data.get('label', '')
            y_offset = data.get('y_offset', 0.0)

            trigger_protocol = None
            offset_val = 0.0

            if label == "speedbump":
                trigger_protocol = "speedbump"
                offset_val = 0.0
            elif label == "small_bump" and abs(y_offset) < 0.10:
                trigger_protocol = "small_bump"
                offset_val = -0.4 if y_offset > 0 else 0.4
            elif label == "crack" and abs(y_offset) < 0.08:
                trigger_protocol = "crack"
                offset_val = -0.3 if y_offset > 0 else 0.3

            if trigger_protocol is not None:
                self.get_logger().info(f'Hazard: {label}! Entering BRACE {trigger_protocol} protocol.')
                self.state = STATE_BRACE
                self.brace_protocol = trigger_protocol
                self.brace_start_x = self.current_x
                self.brace_start_y = self.current_y
                self.steering_offset = offset_val
                self.publish_cmd()

        except Exception as e:
            self.get_logger().error(f"Failed to parse hazard warning: {e}")

    def nav_callback(self, msg):
        self.last_nav_msg = msg
        self.last_nav_time = time.time()
        self.publish_cmd()

    def vidar_callback(self, msg):
        self.last_vidar_msg = msg
        self.last_vidar_time = time.time()
        self.publish_cmd()

    def watchdog_timer_callback(self):
        # Global override: if no cmd_vel_nav for > 0.5s, hard stop
        if time.time() - self.last_nav_time > 0.5:
            stop_msg = Twist()
            stop_msg.linear.x = 0.0
            stop_msg.angular.z = 0.0
            self.cmd_pub.publish(stop_msg)

    def publish_cmd(self):
        # We also need to check watchdog here just in case we are trying to publish from a delayed callback
        if time.time() - self.last_nav_time > 0.5:
            stop_msg = Twist()
            stop_msg.linear.x = 0.0
            stop_msg.angular.z = 0.0
            self.cmd_pub.publish(stop_msg)
            return

        if self.last_nav_msg is None:
            return

        # Gated Priority Multiplexer Logic
        base_vx = self.last_nav_msg.linear.x
        base_wz = self.last_nav_msg.angular.z

        # Check Vidar rules
        if self.last_vidar_msg is not None:
            time_since_vidar = time.time() - self.last_vidar_time

            # Rule 2: Active Brake & Align Reflex (Fresh and Active)
            if self.last_vidar_msg.linear.x >= 0.0 and time_since_vidar < 0.5:
                # Active tracking: take 100% speed, 70/30 steering blend
                base_vx = self.last_vidar_msg.linear.x
                base_wz = (0.7 * self.last_vidar_msg.angular.z) + (0.3 * self.last_nav_msg.angular.z)

            # Rule 3: Latency Safety Net
            # If Vidar sent active command but it's older than 0.8s, fall back to Nav (already handled by default base values)

            # Rule 1: Passive State
            # If linear.x == -1.0, we just use the default base values (100% Nav)

        out_msg = Twist()

        if self.state == STATE_NORMAL:
            out_msg.linear.x = base_vx
            out_msg.angular.z = base_wz

        elif self.state == STATE_BRACE:
            if self.brace_protocol == "speedbump":
                out_msg.linear.x = min(base_vx, 0.10)
                out_msg.angular.z = base_wz
            elif self.brace_protocol == "small_bump":
                out_msg.linear.x = min(base_vx, 0.20)
                out_msg.angular.z = base_wz + self.steering_offset
            elif self.brace_protocol == "crack":
                out_msg.linear.x = min(base_vx, 0.15)
                out_msg.angular.z = base_wz + self.steering_offset
            else:
                out_msg.linear.x = base_vx
                out_msg.angular.z = base_wz

            out_msg.angular.z = self.clamp_steering(out_msg.angular.z)

        self.cmd_pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = AidaBraceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
