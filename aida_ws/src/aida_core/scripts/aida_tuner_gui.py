#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.srv import SetParameters
import tkinter as tk
from tkinter import ttk
import threading

class TuningNode(Node):
    def __init__(self):
        super().__init__('aida_tuning_gui')
        # Create an async client for the parameter service of aida_nav_node
        self.client = self.create_client(SetParameters, '/aida_nav_node/set_parameters')

    def set_parameter(self, param_name, param_value):
        if not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Parameter service not available')
            return

        request = SetParameters.Request()
        param = Parameter(param_name, Parameter.Type.DOUBLE, param_value).to_parameter_msg()
        request.parameters = [param]

        # Call asynchronously to not block the UI
        future = self.client.call_async(request)
        future.add_done_callback(self.parameter_set_callback)

    def parameter_set_callback(self, future):
        try:
            response = future.result()
            for res in response.results:
                if not res.successful:
                    self.get_logger().error(f'Failed to set parameter: {res.reason}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')


class TunerApp:
    def __init__(self, root, ros_node):
        self.root = root
        self.ros_node = ros_node
        self.root.title("AIDA Tuning Dashboard")
        self.root.geometry("400x350")

        self.create_slider("Steering Kp", "kp", 0.0, 1.0, 0.15)
        self.create_slider("Steering Kd", "kd", 0.0, 1.0, 0.10)
        self.create_slider("Gimbal Kp", "gimbal_kp", 0.0, 1.0, 0.20)
        self.create_slider("Max Speed (v_max)", "v_max", 0.0, 1.0, 0.22)

    def create_slider(self, label_text, param_name, min_val, max_val, default_val):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.X)

        label = ttk.Label(frame, text=f"{label_text}: {default_val:.2f}", width=25)
        label.pack(side=tk.LEFT)

        def on_slider_change(val):
            float_val = float(val)
            label.config(text=f"{label_text}: {float_val:.2f}")
            self.ros_node.set_parameter(param_name, float_val)

        slider = ttk.Scale(
            frame,
            from_=min_val,
            to=max_val,
            orient=tk.HORIZONTAL,
            command=on_slider_change
        )
        slider.set(default_val)
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)


def ros_spin_thread(node):
    rclpy.spin(node)


def main():
    rclpy.init()
    tuning_node = TuningNode()

    # Spin ROS 2 node in a background thread so the Tkinter main loop is not blocked
    spin_thread = threading.Thread(target=ros_spin_thread, args=(tuning_node,), daemon=True)
    spin_thread.start()

    # Start Tkinter app
    root = tk.Tk()
    app = TunerApp(root, tuning_node)

    # Run the Tkinter main loop
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        tuning_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
