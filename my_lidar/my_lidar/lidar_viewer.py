#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
import matplotlib.pyplot as plt
import numpy as np
import math

class LidarVisualizer(Node):
    def __init__(self):
        super().__init__('lidar_visualizer')

        self.ranges = None

        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data)

        # Matplotlib 설정 (고정 스케일)
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)

        # 🔴 여기만 변경 (plot → scatter)
        self.lidar_points = self.ax.scatter([], [], s=5)

        # 차량 중심
        self.ax.plot(0, 0, 'ro')

        # 전방 방향 (위쪽)
        self.ax.plot([0, 0], [0, 2], 'r-')

        # 콘 감지 영역 경계선 계산 및 플롯 (초록색 점선 - 반원 형태)
        # 반경 4.0m 이내, 전방 180도
        theta = np.linspace(-np.pi/2, np.pi/2, 100)
        boundary_x = 4.0 * np.sin(theta)
        boundary_y = 4.0 * np.cos(theta)
        self.ax.plot(boundary_x, boundary_y, 'g--', label='Cone Detect Area')
        self.ax.plot([-4.0, 4.0], [0, 0], 'g--')
        self.ax.legend()
        
        self.cone_plots = []

        plt.ion()
        plt.show()

        self.create_timer(0.2, self.timer_callback)

    def lidar_callback(self, msg):
        self.ranges = msg.ranges

    def timer_callback(self):
        if self.ranges is None:
            self.get_logger().warn("No LiDAR data yet")
            return

        ranges = self.ranges

        # NaN 처리 
        valid = np.array([
            d if math.isfinite(d) else np.nan
            for d in ranges
        ])

        # 각도 보정 (0=왼쪽, 90=전방)
        angles = np.deg2rad(np.arange(len(valid)) - 90)

        x = -valid * np.cos(angles)
        y = -valid * np.sin(angles)

        indices = np.arange(len(valid))
        colors = np.full(len(valid), 'b', dtype=object)  # 기본: 파란색

        # 색상 구간 설정
        colors[(indices >= 0) & (indices < 45)] = 'r'           # 🔴 빨강
        colors[(indices >= 45) & (indices < 90)] = 'g'          # 🟢 초록
        colors[(indices >= 90) & (indices < 270)] = 'b'         # 🔵 파랑
        colors[(indices >= 270) & (indices < 315)] = 'orange'   # 🟠 주황
        colors[(indices >= 315) & (indices < 360)] = 'purple'   # 🟣 보라

        # 🔴 scatter 업데이트 (set_data → set_offsets)
        self.lidar_points.set_offsets(np.c_[x, y])
        self.lidar_points.set_color(colors)

        # 이전 검출된 콘 동그라미 표식 제거
        for p in self.cone_plots:
            try:
                p.remove()
            except Exception:
                pass
        self.cone_plots = []

        # 콘 검출 및 동그라미 그리기 (필터: 전방 180도, 반경 4.0m 이내 반원 영역)
        points = []
        for i in range(len(valid)):
            dist = valid[i]
            if 0.1 < dist < 4.0 and math.isfinite(dist):
                angle_deg = i * (360.0 / len(valid))
                if angle_deg > 180:
                    angle_deg -= 360
                
                # 전방 180도 (좌우 90도)
                if -90 <= angle_deg <= 90:
                    # lidar_viewer의 x, y 그리기 방식과 동일한 좌표 연산 수행
                    angle_rad = np.deg2rad(i - 90)
                    xp = -dist * np.cos(angle_rad)
                    yp = -dist * np.sin(angle_rad)
                    points.append((xp, yp))

        if points:
            # DBSCAN 스타일 클러스터링
            clusters = []
            for p in points:
                placed = False
                for c in clusters:
                    rep = c[0]
                    d = math.sqrt((p[0] - rep[0])**2 + (p[1] - rep[1])**2)
                    if d < 0.25: # 25cm 이내 동일한 객체
                        c.append(p)
                        placed = True
                        break
                if not placed:
                    clusters.append([p])
            
            # 최소 2개 포인트 이상인 군집만 라바콘으로 간주
            valid_cones = [c for c in clusters if len(c) >= 2]
            
            # 검출된 각 콘의 중심에 빨간색 동그라미 마커 표시
            for c in valid_cones:
                cx = np.mean([p[0] for p in c])
                cy = np.mean([p[1] for p in c])
                
                cone_plot, = self.ax.plot(cx, cy, 'ro', markersize=15, fillstyle='none', markeredgewidth=2)
                self.cone_plots.append(cone_plot)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        # 전방 거리
        front_candidates = [
            d for d in ranges[85:95]
            if math.isfinite(d)
        ]

        if not front_candidates:
            return

        front = min(front_candidates)
        step = max(1, len(ranges) // 36)

        sample = [
            f"{ranges[i]:.2f}"
            for i in range(0, len(ranges), step)
            if math.isfinite(ranges[i])
        ]

        self.get_logger().info(
            f"Front: {front:.2f} m | Sample: {sample}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = LidarVisualizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()