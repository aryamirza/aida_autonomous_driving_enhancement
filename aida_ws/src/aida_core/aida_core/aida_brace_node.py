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

        # Publishers and Subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.nav_sub = self.create_subscription(Twist, '/cmd_vel_nav', self.nav_callback, 10)
        self.hazard_sub = self.create_subscription(String, '/aida/hazard_warning', self.hazard_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # State variables
        self.state = STATE_NORMAL
        self.last_nav_msg = None
        self.last_nav_time = time.time()

        # Brace state variables
        self.brace_protocol = None
        self.brace_start_time = 0.0
        self.brace_start_x = 0.0
        self.brace_start_y = 0.0
        self.current_x = 0.0
        self.current_y = 0.0
        self.swerve_angular_z = 0.0

        # Watchdog timer
        self.create_timer(0.05, self.watchdog_timer_callback)

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        if self.state == STATE_BRACE and self.brace_protocol == "speedbump":
            dist = math.sqrt((self.current_x - self.brace_start_x)**2 + (self.current_y - self.brace_start_y)**2)
            if dist >= 0.40:
                self.get_logger().info('Speedbump cleared, returning to normal.')
                self.state = STATE_NORMAL
                self.brace_protocol = None
                self.publish_cmd()

    def hazard_callback(self, msg):
        try:
            data = json.loads(msg.data)
            label = data.get('label', '')
            y_offset = data.get('y_offset', 0.0)

            if label == "speedbump":
                self.get_logger().info('Hazard: speedbump! Entering BRACE protocol.')
                self.state = STATE_BRACE
                self.brace_protocol = "speedbump"
                self.brace_start_x = self.current_x
                self.brace_start_y = self.current_y
                self.publish_cmd()

            elif label == "crack" and abs(y_offset) < 0.08:
                self.get_logger().info('Hazard: crack! Entering BRACE swerve protocol.')
                self.state = STATE_BRACE
                self.brace_protocol = "swerve"
                self.brace_start_time = time.time()
                if y_offset > 0:
                    self.swerve_angular_z = -0.3
                else:
                    self.swerve_angular_z = 0.3
                self.publish_cmd()

        except Exception as e:
            self.get_logger().error(f"Failed to parse hazard warning: {e}")

    def nav_callback(self, msg):
        self.last_nav_msg = msg
        self.last_nav_time = time.time()
        self.publish_cmd()

    def watchdog_timer_callback(self):
        # Global override: if no cmd_vel_nav for > 0.5s, hard stop
        if time.time() - self.last_nav_time > 0.5:
            # We don't change state, but we override output to 0.0
            stop_msg = Twist()
            stop_msg.linear.x = 0.0
            stop_msg.angular.z = 0.0
            self.cmd_pub.publish(stop_msg)
            return

        # Handle swerve timeout
        if self.state == STATE_BRACE and self.brace_protocol == "swerve":
            if time.time() - self.brace_start_time >= 0.25:
                self.get_logger().info('Swerve cleared, returning to normal.')
                self.state = STATE_NORMAL
                self.brace_protocol = None
                self.publish_cmd()

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

        out_msg = Twist()

        if self.state == STATE_NORMAL:
            out_msg.linear.x = self.last_nav_msg.linear.x
            out_msg.angular.z = self.last_nav_msg.angular.z

        elif self.state == STATE_BRACE:
            if self.brace_protocol == "speedbump":
                out_msg.linear.x = 0.15
                out_msg.angular.z = self.last_nav_msg.angular.z
            elif self.brace_protocol == "swerve":
                out_msg.linear.x = min(self.last_nav_msg.linear.x, 0.25)
                out_msg.angular.z = self.swerve_angular_z

        self.cmd_pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = AidaBraceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
