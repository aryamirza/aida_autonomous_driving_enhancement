#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Twist
from ros_robot_controller_msgs.msg import MotorsState, MotorState, SetPWMServoState, PWMServoState

class HardwareBridgeNode(Node):
    def __init__(self):
        super().__init__('aida_hardware_bridge')

        # --- Parameters ---
        # Motor parameters
        self.declare_parameter('motor_scale_factor', 1.0)
        self.declare_parameter('invert_right_motors', True)
        self.declare_parameter('right_motor_ids', [2, 4])

        # Steering parameters
        self.declare_parameter('steering_servo_id', 5)
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

        self.get_logger().info('HardwareBridgeNode initialized.')

    def publish_zero_rps(self):
        """Publishes 0.0 rps to all 4 motors."""
        msg = MotorsState()
        for i in range(1, 5):
            motor = MotorState()
            motor.id = i
            motor.rps = 0.0
            msg.data.append(motor)
        self.motor_pub.publish(msg)

    def cmd_vel_callback(self, msg: Twist):
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
        base_rps = linear_x * motor_scale_factor

        motor_msg = MotorsState()
        for i in range(1, 5):
            motor = MotorState()
            motor.id = i

            rps = base_rps
            if invert_right_motors and (i in right_motor_ids):
                rps *= -1.0

            motor.rps = float(rps)
            motor_msg.data.append(motor)

        self.motor_pub.publish(motor_msg)

        # --- Steering ---
        target_pwm = int(pwm_center + (angular_z * steering_scale))
        target_pwm = int(np.clip(target_pwm, pwm_min, pwm_max))

        servo_msg = SetPWMServoState()
        servo_msg.duration = 0.1

        servo = PWMServoState()
        servo.id = [int(steering_servo_id)]
        servo.position = [int(target_pwm)]
        servo.offset = [0]

        servo_msg.state = [servo]

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
