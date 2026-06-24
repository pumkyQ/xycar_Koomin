#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from xycar_msgs.msg import XycarMotor
import os
import sys

class SteerViewerNode(Node):
    def __init__(self):
        super().__init__('steer_viewer')
        self.sub_motor = self.create_subscription(
            XycarMotor, '/xycar_motor', self.motor_callback, 10)
        self.get_logger().info("Steer Viewer Node started. Subscribed to /xycar_motor")

    def motor_callback(self, msg):
        angle = msg.angle
        speed = msg.speed

        # ASCII 가로 게이지바 생성
        # angle 범위: -100(좌) ~ 100(우)
        bar_width = 20
        # -100 ~ 100 -> 0 ~ 20 매핑
        center_idx = bar_width // 2
        active_idx = int(center_idx + (angle / 100.0) * center_idx)
        active_idx = max(0, min(bar_width, active_idx))

        bar = []
        for i in range(bar_width + 1):
            if i == center_idx:
                bar.append('|')
            elif i == active_idx:
                bar.append('▲')
            else:
                bar.append('-')
        bar_str = "".join(bar)

        # 터미널 화면 갱신 출력 (ANSI Escape Code 사용)
        sys.stdout.write('\r')
        sys.stdout.write(
            f"[Steer Gauge] [ {bar_str} ]  |  "
            f"Steer Angle (Motor): {angle:+.1f}°  |  "
            f"Speed: {speed:.1f}  "
        )
        sys.stdout.flush()

def main(args=None):
    rclpy.init(args=args)
    node = SteerViewerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nSteer Viewer Node terminated.")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
