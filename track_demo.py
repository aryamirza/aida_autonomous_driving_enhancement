#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from ros_robot_controller_msgs.msg import MotorsState, MotorState, SetPWMServoState, PWMServoState
import sys
import termios
import tty
import threading
import time
import numpy as np

class TrackDemoNode(Node):
    def __init__(self):
        super().__init__('track_demo_node')

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

        self.is_running = False
        self.stop_requested = False

        self.motor_scale_factor = 4.9 # from memory
        self.pwm_center = 1500
        self.steering_scale = 400.0 # From memory

        self.get_logger().info('Track Demo Node started. Press "Q" to start, "W" to stop.')

        self.keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_thread.start()

        self.sequence_thread = None

    def keyboard_listener(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while True:
                char = sys.stdin.read(1).lower()
                if char == 'q' and not self.is_running:
                    self.get_logger().info('Start requested. Commencing track sequence...')
                    self.is_running = True
                    self.stop_requested = False
                    self.sequence_thread = threading.Thread(target=self.run_sequence, daemon=True)
                    self.sequence_thread.start()
                elif char == 'w' and self.is_running:
                    self.get_logger().info('Stop requested. Halting vehicle...')
                    self.stop_requested = True
                    self.is_running = False
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def publish_cmd(self, linear_x, angular_z):
        # Throttle
        base_rps = float(linear_x * self.motor_scale_factor)
        motor_msg = MotorsState()

        # Left Motor (ID 4) - Inverted
        m_left = MotorState()
        m_left.id = 4
        m_left.rps = base_rps * -1.0

        # Right Motor (ID 2) - Positive
        m_right = MotorState()
        m_right.id = 2
        m_right.rps = base_rps * 1.0

        motor_msg.data = [m_left, m_right]
        self.motor_pub.publish(motor_msg)

        # Steering
        target_pwm = int(self.pwm_center + (angular_z * self.steering_scale))
        target_pwm = int(np.clip(target_pwm, 1100, 1900))

        servo_msg = SetPWMServoState()
        servo_msg.duration = 0.1

        servo_steer = PWMServoState()
        servo_steer.id = [3]  # ID 3 is the steering rack
        servo_steer.position = [target_pwm]
        servo_steer.offset = [0]

        servo_msg.state = [servo_steer]
        self.servo_pub.publish(servo_msg)

    def drive(self, speed, angular, duration):
        """Drives the car at a set speed and steering angle for a duration."""
        start_time = time.time()
        while time.time() - start_time < duration:
            if self.stop_requested:
                self.publish_cmd(0.0, 0.0)
                return False
            self.publish_cmd(speed, angular)
            time.sleep(0.05)
        return True

    def run_sequence(self):
        # Continuous sequence loop
        lap = 1
        while self.is_running and not self.stop_requested:
            self.get_logger().info(f'--- Starting Lap {lap} ---')

            # --- START ON RIGHT LANE (x=30cm) ---
            # Speed bump is at x=70cm. Car front is at 30cm. Distance to bump = 40cm.
            # We drive normally for a bit, then slow down before the bump.
            # Normal speed: 0.2 m/s. Slower speed: 0.1 m/s.

            # Drive straight 25cm (0.25m) at 0.2 m/s -> 1.25s
            if not self.drive(0.2, 0.0, 1.25): break

            # Slow down before reaching bump, cross it, and pass it.
            # Drive 30cm (0.30m) at 0.1 m/s -> 3.0s
            if not self.drive(0.1, 0.0, 3.0): break

            # Resume normal speed to end of right lane.
            # Total track length 120cm. We covered 30cm(start) + 25cm + 30cm = 85cm.
            # Remaining length = 35cm (0.35m).
            # Drive 35cm at 0.2 m/s -> 1.75s
            if not self.drive(0.2, 0.0, 1.75): break

            # --- TURN TO LEFT LANE ---
            # Left U-Turn semi-circle. Positive angular is left.
            # Assuming radius of 20cm, circumference of semi-circle = pi * r = 0.628m.
            # At 0.2m/s, it takes ~3.14s. Angular velocity = v/r = 0.2/0.2 = 1.0 rad/s.
            if not self.drive(0.2, 1.0, 3.14): break

            # --- ON LEFT LANE, HEADING BACK ---
            # Total length = 120cm.
            # Obstacle 1 (Small Bump): 35cm from the OTHER end (so 35cm from where we are now).
            # We need to deviate before it, maintaining normal speed.

            # Drive straight 20cm (0.20m) -> 1.0s
            if not self.drive(0.2, 0.0, 1.0): break

            # Deviate Right to avoid the bump.
            # Nudge right (angular < 0) then left to straighten.
            if not self.drive(0.2, -0.6, 0.5): break # Nudge right
            if not self.drive(0.2, 0.6, 0.5): break  # Correct left

            # Drive past the bump
            if not self.drive(0.2, 0.0, 1.0): break  # 20cm straight

            # Return to center lane
            if not self.drive(0.2, 0.6, 0.5): break  # Nudge left
            if not self.drive(0.2, -0.6, 0.5): break # Correct right

            # Obstacle 2 (Cracks): 40cm from the starting end.
            # Since we started from x=120 and are driving back, the cracks are at x=40cm relative to the track start.
            # That means it's 120 - 40 = 80cm from the end we just turned from.
            # We have covered roughly 20 + 20(deviating) + 20 + 20 = 80cm! So we are right at the cracks.
            # We must slow down to 0.1 m/s and deviate LEFT (the "less cracked" area is on the left side).

            # Deviate Left AND Slow Down
            # We have 40cm left on the track
            if not self.drive(0.1, 0.6, 0.5): break   # Nudge left while slow (5cm)
            if not self.drive(0.1, -0.6, 0.5): break  # Correct right while slow (5cm)

            # Drive past the cracks
            if not self.drive(0.1, 0.0, 1.0): break   # 10cm straight

            # Return to center lane and speed up
            if not self.drive(0.2, -0.6, 0.5): break  # Nudge right fast (10cm)
            if not self.drive(0.2, 0.6, 0.5): break   # Correct left fast (10cm)

            # Total remaining distance covered: 5 + 5 + 10 + 10 + 10 = 40cm. We are at the end.

            # --- TURN TO RIGHT LANE ---
            # Left U-Turn back to the start lane.
            if not self.drive(0.2, 1.0, 3.14): break

            # We are back on the right lane, but at x=0. The start is at x=30cm.
            # Drive 30cm at 0.2m/s to reach the starting position.
            if not self.drive(0.2, 0.0, 1.5): break

            lap += 1

        self.publish_cmd(0.0, 0.0)
        self.is_running = False

def main(args=None):
    rclpy.init(args=args)
    node = TrackDemoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_cmd(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
