#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from ros_robot_controller_msgs.msg import MotorsState, MotorState, SetPWMServoState, PWMServoState
import sys
import termios
import tty
import threading

class HardwareBridgeNode(Node):
    def __init__(self):
        super().__init__('aida_hardware_bridge')

        self.is_engaged = False
        self.max_linear_speed = 0.5

        # --- Parameters ---
        # Motor parameters
        self.declare_parameter('motor_scale_factor', 4.9)
        self.declare_parameter('invert_right_motors', True)
        self.declare_parameter('right_motor_ids', [2, 4])

        # Steering parameters
        self.declare_parameter('steering_servo_id', 3)
        self.declare_parameter('pwm_center', 1500)
        self.declare_parameter('steering_scale', 400.0)
        self.declare_parameter('pwm_max', 1900)
        self.declare_parameter('pwm_min', 1100)

        # --- Publishers & Subscribers ---
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/ros_robot_controller/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        self.gimbal_sub = self.create_subscription(
            JointState,
            '/camera/gimbal_cmd',
            self.gimbal_callback,
            10
        )

        self.motor_pub = self.create_publisher(
            MotorsState,
            '/ros_robot_controller/set_motor',
            10
        )

        self.servo_pub = self.create_publisher(
            SetPWMServoState,
            '/ros_robot_controller/pwm_servo/set_state',
            10
        )

        # --- Watchdog Timer ---
        self.watchdog_timeout = 0.5
        self.watchdog_timer = self.create_timer(self.watchdog_timeout, self.watchdog_callback)
        self.last_cmd_time = self.get_clock().now()

        # Start keyboard listener thread
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_thread.start()

        # Init camera stance
        self.init_camera_timer = self.create_timer(0.5, self.init_camera_stance)

        self.get_logger().info('HardwareBridgeNode initialized.')

    def init_camera_stance(self):
        """Initialize the camera to standby stance: Pan=1500, Tilt=1500."""
        self.init_camera_timer.cancel() # Run once

        servo_msg = SetPWMServoState()
        servo_msg.duration = 0.5

        servo_tilt = PWMServoState()
        servo_tilt.id = [1]
        servo_tilt.position = [1500]
        servo_tilt.offset = [0]

        servo_pan = PWMServoState()
        servo_pan.id = [4]
        servo_pan.position = [1500]
        servo_pan.offset = [0]

        servo_msg.state = [servo_tilt, servo_pan]
        self.servo_pub.publish(servo_msg)
        self.get_logger().info('Camera stance initialized (Pan: 1500, Tilt: 1500)')

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
                    self.get_logger().info('[INFO] HARDWARE UNLOCKED - LISTENING TO COMMANDS')
                elif char == 'a' and self.is_engaged:
                    self.is_engaged = False
                    self.get_logger().warn('[WARN] HARDWARE E-STOP - MOTORS LOCKED')
                    self.publish_zero_rps()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def gimbal_callback(self, msg: JointState):
        """Processes AI Pilot JointState messages and outputs SetPWMServoState messages."""
        if not self.is_engaged:
            return

        if len(msg.position) >= 2:
            pan_rad = msg.position[0]
            tilt_rad = msg.position[1]

            pan_pwm = int(1500 + (pan_rad * 500))
            tilt_pwm = int(1500 - (tilt_rad * 500))
            tilt_pwm = max(1000, min(2000, tilt_pwm))

            servo_msg = SetPWMServoState()
            servo_msg.duration = 0.1

            servo_tilt = PWMServoState()
            servo_tilt.id = [1]
            servo_tilt.position = [tilt_pwm]
            servo_tilt.offset = [0]

            servo_pan = PWMServoState()
            servo_pan.id = [4]
            servo_pan.position = [pan_pwm]
            servo_pan.offset = [0]

            servo_msg.state = [servo_tilt, servo_pan]
            self.servo_pub.publish(servo_msg)


    def publish_zero_rps(self):
        """Publishes 0.0 rps to the active motors (M4 and M2)."""
        msg = MotorsState()

        m_left = MotorState()
        m_left.id = 4
        m_left.rps = 0.0

        m_right = MotorState()
        m_right.id = 2
        m_right.rps = 0.0

        msg.data = [m_left, m_right]
        self.motor_pub.publish(msg)

    def cmd_vel_callback(self, msg: Twist):
        if not self.is_engaged:
            return

        # Reset watchdog timer
        self.last_cmd_time = self.get_clock().now()

        linear_x = msg.linear.x
        angular_z = msg.angular.z

        # Safety Watchdog Event-Driven (Zero Command)
        if linear_x == 0.0 and angular_z == 0.0:
            self.publish_zero_rps()
            # Do not publish steering command to hold the angle
            return

        # Fetch current parameters
        motor_scale_factor = self.get_parameter('motor_scale_factor').value
        invert_right_motors = self.get_parameter('invert_right_motors').value
        right_motor_ids = self.get_parameter('right_motor_ids').value

        steering_servo_id = self.get_parameter('steering_servo_id').value
        pwm_center = self.get_parameter('pwm_center').value
        steering_scale = self.get_parameter('steering_scale').value
        pwm_max = self.get_parameter('pwm_max').value
        pwm_min = self.get_parameter('pwm_min').value

        # --- Throttle ---
        base_rps = float(linear_x * motor_scale_factor)

        motor_msg = MotorsState()

        # Left Motor (ID 4) - Must be INVERTED to physically drive forward
        m_left = MotorState()
        m_left.id = 4
        m_left.rps = base_rps * -1.0

        # Right Motor (ID 2) - Must be POSITIVE to physically drive forward
        m_right = MotorState()
        m_right.id = 2
        m_right.rps = base_rps * 1.0

        motor_msg.data = [m_left, m_right]
        self.motor_pub.publish(motor_msg)

        # --- Steering ---
        target_pwm = int(pwm_center + (angular_z * steering_scale))
        target_pwm = int(np.clip(target_pwm, pwm_min, pwm_max))

        servo_msg = SetPWMServoState()
        servo_msg.duration = 0.1

        # 4. SERVO BUNDLE (Steering only)

        # Steering (J3) - Physical front wheel rack
        servo_steer = PWMServoState()
        servo_steer.id = [3]  # Note: assuming physical rack is id=3 per earlier snippet, but we can also use int(steering_servo_id)
        servo_steer.position = [int(target_pwm)]
        servo_steer.offset = [0]

        servo_msg.state = [servo_steer]

        self.servo_pub.publish(servo_msg)

    def watchdog_callback(self):
        """Timer-based watchdog that stops the motors if no command is received."""
        now = self.get_clock().now()
        time_since_last_cmd = (now - self.last_cmd_time).nanoseconds / 1e9

        if time_since_last_cmd >= self.watchdog_timeout:
            self.publish_zero_rps()
            # Only log once until we get a new command
            if time_since_last_cmd < self.watchdog_timeout * 2:
                 self.get_logger().warn(f'Watchdog timeout ({self.watchdog_timeout}s exceeded)! Coasting prevented. Motors halted.')

def main(args=None):
    rclpy.init(args=args)
    node = HardwareBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
