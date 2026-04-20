#!/usr/bin/env python3
"""Launch YOLO lidar consumer and RViz2 with image panels."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory('lidarscan_msg_yolo_demo')
    rviz_cfg = os.path.join(pkg_share, 'config', 'lidarscan_view.rviz')

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'model_name',
                default_value='yolo11n.pt',
                description='Ultralytics YOLO weights (built-in name or path to .pt)',
            ),
            DeclareLaunchArgument(
                'confidence',
                default_value='0.25',
                description='Minimum confidence for detections',
            ),
            DeclareLaunchArgument(
                'device',
                default_value='',
                description='Torch device, e.g. cuda:0 or cpu (empty = auto)',
            ),
            DeclareLaunchArgument(
                'use_rviz',
                default_value='true',
                description='If true, start RViz2 with yolo_lidar_view.rviz',
            ),
            Node(
                package='lidarscan_msg_yolo_demo',
                executable='lidarscan_yolo_consumer',
                name='lidarscan_yolo_consumer',
                output='screen',
                parameters=[
                    {'model_name': LaunchConfiguration('model_name')},
                    {
                        'confidence': ParameterValue(
                            LaunchConfiguration('confidence'), value_type=float
                        )
                    },
                    {'device': LaunchConfiguration('device')},
                ],
            ),
            ExecuteProcess(
                cmd=['rviz2', '-d', rviz_cfg],
                output='screen',
                condition=IfCondition(LaunchConfiguration('use_rviz')),
            ),
        ]
    )
