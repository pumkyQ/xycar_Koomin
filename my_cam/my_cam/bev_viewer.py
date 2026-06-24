#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
from xycar_msgs.msg import XycarMotor
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge

class BEVViewerNode(Node):
    def __init__(self):
        super().__init__('bev_viewer')
        self.bridge = CvBridge()
        self.image = None
        self.current_speed = 0.0

        # Subscriber
        self.sub_front = self.create_subscription(
            Image, '/usb_cam/image_raw/front', self.img_callback, qos_profile_sensor_data)
        
        self.sub_motor = self.create_subscription(
            XycarMotor, '/xycar_motor', self.motor_callback, 10)

        # Timer (30 FPS)
        self.timer = self.create_timer(0.033, self.process_image)
        self.get_logger().info("BEV Viewer Node started. Subscribed to /usb_cam/image_raw/front & /xycar_motor")

    def img_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def motor_callback(self, msg):
        self.current_speed = msg.speed

    def process_image(self):
        if self.image is None:
            return

        img = self.image.copy()
        h, w = img.shape[:2]

        # 1. 원본 이미지에 BEV 변환 소스 점(Trapezoid) 그리기
        src_pts = np.float32([
            [w * 0.25, h * 0.55],  # 좌상단
            [w * 0.75, h * 0.55],  # 우상단
            [w * 0.05, h * 0.95],  # 좌하단
            [w * 0.95, h * 0.95]   # 우하단
        ])

        # 원본 이미지에 소스 영역 다각형 그리기 (초록색)
        draw_pts = src_pts.astype(np.int32)
        cv2.polylines(img, [draw_pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # 2. 투영 매트릭스 계산 및 변환 (BEV)
        dst_pts = np.float32([
            [w * 0.2, 0],
            [w * 0.8, 0],
            [w * 0.2, h],
            [w * 0.8, h]
        ])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        bev_img = cv2.warpPerspective(self.image, M, (w, h))

        # 3. BEV 이미지에 HSV 색 필터링 적용 (lane_detector.py 기준)
        hsv_bev = cv2.cvtColor(bev_img, cv2.COLOR_BGR2HSV)

        # 흰색 차선 필터링
        white_mask = cv2.inRange(hsv_bev, np.array([0, 0, 200]), np.array([180, 40, 255]))
        
        # 노란색 차선 필터링
        yellow_mask = cv2.inRange(hsv_bev, np.array([15, 80, 150]), np.array([35, 255, 255]))
        
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

        # 모폴로지 연산으로 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # 4. 차선 중심 추적 (슬라이딩 윈도우 시각화용)
        # 히스토그램으로 좌우 차선 시작점 추정
        histogram = np.sum(combined_mask[h // 2:, :], axis=0)
        midpoint = w // 2
        left_base = np.argmax(histogram[:midpoint]) if np.max(histogram[:midpoint]) > 0 else w // 4
        right_base = np.argmax(histogram[midpoint:]) + midpoint if np.max(histogram[midpoint:]) > 0 else w * 3 // 4

        left_current = left_base
        right_current = right_base

        left_pts = []
        right_pts = []
        center_pts = []

        window_h = h // 10
        for i in range(10):
            y_low = h - (i + 1) * window_h
            y_high = h - i * window_h
            y_center = (y_low + y_high) // 2

            # 좌측 차선 검출 및 업데이트
            left_area = combined_mask[y_low:y_high, max(0, left_current - 45):min(w, left_current + 45)]
            if np.sum(left_area) > 100:
                left_current = int(left_current - 45 + np.mean(np.where(left_area > 0)[1]))
            
            # 우측 차선 검출 및 업데이트
            right_area = combined_mask[y_low:y_high, max(0, right_current - 45):min(w, right_current + 45)]
            if np.sum(right_area) > 100:
                right_current = int(right_current - 45 + np.mean(np.where(right_area > 0)[1]))

            left_pts.append((left_current, y_center))
            right_pts.append((right_current, y_center))
            center_pts.append(((left_current + right_current) // 2, y_center))

        # 5. Lookahead Distance & 타겟 진행 방향 계산
        # track_drive.py 튜닝 스펙과 동일한 L_d 계산
        lookahead_distance = max(0.5, min(1.8, 0.4 + 0.08 * self.current_speed))
        
        # 거리를 픽셀 좌표(y)로 선형 변핑 (0.5m ~ 1.8m -> h ~ 0)
        target_y = int(h - (lookahead_distance - 0.5) / (1.8 - 0.5) * h)
        target_y = max(0, min(h - 1, target_y))

        # target_y에 해당하는 슬라이딩 윈도우 인덱스
        target_idx = int((h - target_y) / window_h)
        target_idx = max(0, min(9, target_idx))

        target_x = center_pts[target_idx][0]

        # 6. BEV 이미지 위에 시각화 그리기
        # 차선 중심 경로 선 그리기 (초록색)
        for i in range(len(center_pts) - 1):
            cv2.line(bev_img, center_pts[i], center_pts[i+1], (0, 255, 0), 2)
            cv2.circle(bev_img, left_pts[i], 3, (255, 0, 0), -1)   # 좌차선 (파랑)
            cv2.circle(bev_img, right_pts[i], 3, (0, 0, 255), -1)  # 우차선 (빨강)

        # Lookahead boundary (전방주시거리 한계선 - 빨간색 수평선)
        cv2.line(bev_img, (0, target_y), (w, target_y), (0, 165, 255), 1, cv2.LINE_AA)
        
        # 타겟 주행 목표점 (노란색 원)
        cv2.circle(bev_img, (target_x, target_y), 8, (0, 255, 255), -1)

        # 현재 차량 진로 방향 화살표 (보라색 화살표)
        cv2.arrowedLine(bev_img, (w // 2, h - 1), (target_x, target_y), (255, 0, 255), 3, tipLength=0.1)

        # 7. 시각화용 이미지 생성
        mask_bgr = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)

        # 원본(ROI 표시), BEV 변환본(조향 정보 오버레이), 필터링 마스크를 가로로 이어붙임
        combined = np.hstack((img, bev_img, mask_bgr))
        
        # 각 화면 이름 텍스트 출력
        cv2.putText(combined, "1. Front Cam (Green ROI)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, "2. Bird's Eye View (Target Heading)", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, "3. HSV Filtered Mask", (w*2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 상단 오버레이 정보
        cv2.putText(combined, f"Speed: {self.current_speed:.1f} | L_d: {lookahead_distance:.2f}m", (w + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 리사이즈해서 화면 표시
        display_w = int(w * 3 * 0.7)
        display_h = int(h * 0.7)
        combined_resized = cv2.resize(combined, (display_w, display_h))

        cv2.imshow("Lane Detection BEV Viewer", combined_resized)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = BEVViewerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
