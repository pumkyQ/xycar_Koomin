#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
장애물 감지 모듈 (Obstacle Detector)
- LiDAR 데이터를 사용하여 라바콘, 보행자, 차량 등의 장애물을 감지합니다.
- 카메라 영상을 활용하여 보행자와 차량을 구분합니다.
- 장애물 종류에 따른 회피 방향과 거리 정보를 제공합니다.
"""
import cv2
import numpy as np
import math

# ============================================
# LiDAR 설정 상수
# ============================================
LIDAR_TOTAL_POINTS = 360     # 기본 360개 포인트 (1도당 1개)
LIDAR_MAX_RANGE = 10.0       # 최대 감지 거리 (미터)

FRONT_ANGLE_MIN = -30        # 우측 30도 (음수 각도)
FRONT_ANGLE_MAX = 30         # 좌측 30도 (양수 각도)

CONE_DETECT_RANGE = 6.5      # 라바콘 감지 최대 거리 (미터) (늘림)
CONE_MIN_POINTS = 2          # 라바콘 판별 최소 점 수 (줄임)

OBSTACLE_NEAR_RANGE = 2.5    # 근접 장애물 임계 거리 (미터)
OBSTACLE_DANGER_RANGE = 1.5  # 위험 거리 (즉시 정지 필요)
OBSTACLE_FAR_RANGE = 4.0     # 원거리 장애물 감지 거리 (미터)
POLICE_DETECT_RANGE = 5.0    # 경찰차 감지 거리

class ObstacleDetector:
    def __init__(self):
        self.prev_obstacles = []
        self.cone_mode = False
        self.obstacle_history = []
        self.history_max = 5

    def detect_front_obstacle(self, lidar_ranges):
        result = {
            'min_dist': float('inf'),
            'min_angle': 0,
            'left_dist': float('inf'),
            'right_dist': float('inf'),
            'obstacle_detected': False,
            'danger': 'NONE'
        }

        if lidar_ranges is None:
            return result

        ranges = np.array(lidar_ranges, dtype=np.float32)
        n = len(ranges)
        if n == 0:
            return result

        angles_per_point = 360.0 / n
        front_indices = []
        for i in range(n):
            angle = i * angles_per_point
            if angle > 180:
                angle -= 360

            if FRONT_ANGLE_MIN <= angle <= FRONT_ANGLE_MAX:
                front_indices.append(i)

        if not front_indices:
            return result

        front_ranges = ranges[front_indices]
        valid_mask = (front_ranges > 0.1) & (front_ranges < LIDAR_MAX_RANGE) & np.isfinite(front_ranges)

        if not np.any(valid_mask):
            return result

        valid_ranges = front_ranges[valid_mask]
        valid_indices = np.array(front_indices)[valid_mask]

        min_idx = np.argmin(valid_ranges)
        result['min_dist'] = float(valid_ranges[min_idx])
        result['min_angle'] = float(valid_indices[min_idx] * angles_per_point)
        if result['min_angle'] > 180:
            result['min_angle'] -= 360

        left_mask = np.array([i * angles_per_point for i in valid_indices])
        left_mask = np.where(left_mask > 180, left_mask - 360, left_mask)

        # angle > 0 is left, angle < 0 is right
        left_ranges = valid_ranges[left_mask > 0]
        right_ranges = valid_ranges[left_mask <= 0]

        if len(left_ranges) > 0:
            result['left_dist'] = float(np.mean(left_ranges))
        if len(right_ranges) > 0:
            result['right_dist'] = float(np.mean(right_ranges))

        result['obstacle_detected'] = True
        if result['min_dist'] < OBSTACLE_DANGER_RANGE:
            result['danger'] = 'DANGER'
        elif result['min_dist'] < OBSTACLE_NEAR_RANGE:
            result['danger'] = 'NEAR'
        elif result['min_dist'] < OBSTACLE_FAR_RANGE:
            result['danger'] = 'FAR'
        else:
            result['obstacle_detected'] = False
            result['danger'] = 'NONE'

        return result

    def detect_cones(self, lidar_ranges):
        result = {
            'cone_detected': False,
            'steer_offset': 0.0,
            'left_cones': [],
            'right_cones': []
        }

        if lidar_ranges is None:
            return result

        ranges = np.array(lidar_ranges, dtype=np.float32)
        n = len(ranges)
        if n == 0:
            return result

        angles_per_point = 360.0 / n
        left_points = []
        right_points = []

        for i in range(n):
            angle_deg = i * angles_per_point
            if angle_deg > 180:
                angle_deg -= 360

            dist = ranges[i]

            if not (0.1 < dist < CONE_DETECT_RANGE and np.isfinite(dist)):
                continue

            # 극좌표 -> 직교좌표 (x, y) 변환 및 가로 오프셋(x) 기준 도로 바깥 장애물 필터링
            angle_rad = math.radians(angle_deg)
            x = dist * math.sin(angle_rad)
            if abs(x) >= 1.3:
                continue

            # 양수 각도는 좌측, 음수 각도는 우측 (시야 확장)
            if 0 < angle_deg <= 75:
                left_points.append(dist)
            elif -75 <= angle_deg < 0:
                right_points.append(dist)

        if len(left_points) >= CONE_MIN_POINTS and len(right_points) >= CONE_MIN_POINTS:
            result['cone_detected'] = True
            # 가장 가까운 점 3개의 평균만 사용 (먼 배경/벽 필터링)
            left_points.sort()
            right_points.sort()
            left_avg = np.mean(left_points[:3])
            right_avg = np.mean(right_points[:3])

            # 왼쪽이 더 가까우면 (left_avg가 작으면) 오른쪽으로 조향해야 함 (양수)
            # 오른쪽이 더 가까우면 (right_avg가 작으면) 왼쪽으로 조향해야 함 (음수)
            diff = right_avg - left_avg
            result['steer_offset'] = diff * 60.0  # 조향 게인 대폭 상향 (감도 강화)

        elif len(left_points) >= CONE_MIN_POINTS:
            result['cone_detected'] = True
            result['steer_offset'] = 45.0

        elif len(right_points) >= CONE_MIN_POINTS:
            result['cone_detected'] = True
            result['steer_offset'] = -45.0

        return result

    def detect_pedestrian(self, image, lidar_ranges):
        result = {
            'pedestrian_detected': False,
            'pedestrian_dist': float('inf'),
            'pedestrian_direction': 'CENTER',
            'should_stop': False
        }

        if image is None:
            return result

        h, w = image.shape[:2]
        roi = image[int(h * 0.3):int(h * 0.85), :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        dark_mask = cv2.inRange(hsv, np.array([0, 0, 20]), np.array([180, 255, 120]))
        road_mask = cv2.inRange(hsv, np.array([0, 0, 80]), np.array([180, 30, 180]))
        dark_mask = cv2.bitwise_and(dark_mask, cv2.bitwise_not(road_mask))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = bh / max(bw, 1)

            if aspect_ratio > 1.5 and bh > 40:
                result['pedestrian_detected'] = True
                center_x = x + bw // 2
                roi_w = roi.shape[1]
                if center_x < roi_w // 3:
                    result['pedestrian_direction'] = 'LEFT'
                elif center_x > roi_w * 2 // 3:
                    result['pedestrian_direction'] = 'RIGHT'
                else:
                    result['pedestrian_direction'] = 'CENTER'
                break

        if result['pedestrian_detected'] and lidar_ranges is not None:
            front_info = self.detect_front_obstacle(lidar_ranges)
            if front_info['obstacle_detected']:
                result['pedestrian_dist'] = front_info['min_dist']
                if front_info['danger'] in ('DANGER', 'NEAR'):
                    result['should_stop'] = True

        return result

    def detect_vehicle(self, image, lidar_ranges):
        result = {
            'vehicle_detected': False,
            'vehicle_dist': float('inf'),
            'overtake_direction': 'LEFT',
            'should_overtake': False
        }

        if lidar_ranges is None:
            return result

        front_info = self.detect_front_obstacle(lidar_ranges)

        if not front_info['obstacle_detected']:
            return result

        if front_info['min_dist'] < OBSTACLE_FAR_RANGE:
            result['vehicle_detected'] = True
            result['vehicle_dist'] = front_info['min_dist']

            # 장애물이 우측(< 0)이면 좌측으로 추월, 좌측(> 0)이면 우측으로 추월
            if front_info['min_angle'] < 0:
                result['overtake_direction'] = 'LEFT'
            else:
                result['overtake_direction'] = 'RIGHT'

            if front_info['danger'] in ('NEAR', 'DANGER'):
                result['should_overtake'] = True

        return result

    def detect_police_car(self, image):
        if image is None:
            return False

        h, w = image.shape[:2]
        roi = image[int(h * 0.3):int(h * 0.75), :int(w * 0.5)]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        blue_mask = cv2.inRange(hsv, np.array([100, 100, 100]), np.array([130, 255, 255]))
        blue_area = cv2.countNonZero(blue_mask)

        red_mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        red_area = cv2.countNonZero(red_mask1) + cv2.countNonZero(red_mask2)

        if blue_area > 300 and red_area > 300:
            return True

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 3000:
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw > 50 and bh > 30:
                    return True

        return False
