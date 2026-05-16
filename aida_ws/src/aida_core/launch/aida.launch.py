from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='aida_core',
            executable='impact_verifier_node',
            name='impact_verifier_node',
            output='screen'
        ),
        Node(
            package='aida_core',
            executable='aida_vidar_node',
            name='aida_vidar_node',
            output='screen'
        ),
        Node(
            package='aida_core',
            executable='aida_nav_node',
            name='aida_nav_node',
            output='screen'
        ),
        Node(
            package='aida_core',
            executable='aida_memory_node',
            name='aida_memory_node',
            output='screen'
        ),
        Node(
            package='aida_core',
            executable='aida_brace_node',
            name='aida_brace_node',
            output='screen'
        ),
        Node(
            package='aida_core',
            executable='aida_hardware_bridge',
            name='aida_hardware_bridge',
            output='screen',
            remappings=[
                ('/ros_robot_controller/cmd_vel', '/cmd_vel')
            ]
        )
    ])