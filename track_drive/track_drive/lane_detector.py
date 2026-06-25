#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
차선 인식 모듈 (Lane Detector)
- 카메라 영상을 Bird's Eye View (BEV)로 변환한 후,
  속도 기반 동적 전방주시거리(lookahead distance)에 위치한 차선 중심과의 
  물리적 횡방향 오차(e_y, 단위: m)를 계산하여 반환합니다.
- HSV 색공간 기반 필터링과 슬라이딩 윈도우 기법을 사용합니다.
"""
import cv2
import numpy as np

# ============================================
# 상수 정의
# ============================================
IMG_WIDTH = 640
IMG_HEIGHT = 480

# 슬라이딩 윈도우 설정
NUM_WINDOWS = 10           # 세로 방향 윈도우 개수
WINDOW_MARGIN = 75         # 윈도우 좌우 탐색 범위 (픽셀)
MIN_PIX_RECENTER = 30      # 윈도우 재조정 최소 픽셀 수

# HSV 색공간 임계값 - 흰색 차선
WHITE_H_MIN, WHITE_H_MAX = 0, 180
WHITE_S_MIN, WHITE_S_MAX = 0, 40
WHITE_V_MIN, WHITE_V_MAX = 150, 255

# HSV 색공간 임계값 - 노란색 차선
YELLOW_H_MIN, YELLOW_H_MAX = 15, 35
YELLOW_S_MIN, YELLOW_S_MAX = 50, 255
YELLOW_V_MIN, YELLOW_V_MAX = 80, 255

class LaneDetector:
    """
    카메라 영상을 BEV로 변환한 뒤, 동적 전방주시거리의 물리적 오차를 반환합니다.
    """
    def __init__(self):
        # BEV 변환 소스 점(Trapezoid) 정의
        self.src_pts = np.float32([
            [IMG_WIDTH * 0.25, IMG_HEIGHT * 0.55],  # 좌상단
            [IMG_WIDTH * 0.75, IMG_HEIGHT * 0.55],  # 우상단
            [IMG_WIDTH * 0.05, IMG_HEIGHT * 0.95],  # 좌하단
            [IMG_WIDTH * 0.95, IMG_HEIGHT * 0.95]   # 우하단
        ])

        # BEV 변환 목적지 점(Rectangle) 정의
        self.dst_pts = np.float32([
            [IMG_WIDTH * 0.2, 0],
            [IMG_WIDTH * 0.8, 0],
            [IMG_WIDTH * 0.2, IMG_HEIGHT],
            [IMG_WIDTH * 0.8, IMG_HEIGHT]
        ])

        # 투영 변환 행렬 계산
        self.M = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)

        # BEV 가로 픽셀당 물리적 거리 (m/pixel) 변환 계수
        # BEV 상에서 차선간 거리는 w * 0.6 = 384픽셀이며, 실제 차선 폭은 대략 1.0m입니다.
        self.bev_pixel_to_meter = 1.0 / 384.0

        # 이전 프레임의 차선 중심 위치 저장 (차선 미검출 시 보정용)
        self.prev_left_x = int(IMG_WIDTH * 0.2)
        self.prev_right_x = int(IMG_WIDTH * 0.8)
        self.prev_center_x = IMG_WIDTH // 2
        
        # 차선 검출 실패 카운터
        self.no_lane_count = 0
        
        # BEV 상의 차선간 거리 초기값 (동적 추적 학습용, 초기값 240)
        self.lane_width_bev = 240

    def detect(self, image, lookahead_distance):
        """
        카메라 영상을 BEV로 변환하여 동적 Lookahead 거리에서의 물리적 오프셋 e_y(m)를 반환합니다.
        """
        if image is None:
            return 0.0, None

        h, w = image.shape[:2]

        # 1. BEV 변환 투영 적용
        bev_img = cv2.warpPerspective(image, self.M, (w, h))

        # 2. HSV 변환 및 차선 필터링
        hsv_bev = cv2.cvtColor(bev_img, cv2.COLOR_BGR2HSV)
        white_mask = self._filter_white(hsv_bev)
        yellow_mask = self._filter_yellow(hsv_bev)

        # 3. 어두운 도로 검은색 마스크를 통한 잔디/나무 노이즈 차단 (그림자 포함하도록 구간 대폭 확장)
        lower_road = np.array([0, 0, 10])
        upper_road = np.array([180, 80, 200])
        road_mask = cv2.inRange(hsv_bev, lower_road, upper_road)
        
        # 도로 마스크 팽창
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        dilated_road_mask = cv2.dilate(road_mask, kernel_dilate, iterations=1)
        
        # 도로 영역 내부에서 검출된 차선 후보만 유효화
        white_mask = cv2.bitwise_and(white_mask, dilated_road_mask)
        yellow_mask = cv2.bitwise_and(yellow_mask, dilated_road_mask)

        # 모폴로지 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

        # 4. 슬라이딩 윈도우 추적 (노란색/흰색 분리 감지 적용)
        left_pts, right_pts, center_pts, debug_bev = self._sliding_window_bev(
            white_mask, yellow_mask, combined_mask, bev_img.copy()
        )

        # 5. Lookahead 거리에 따른 target_y 좌표 매핑
        # 거리 범위 (0.5m ~ 1.8m) -> 이미지 행 (h ~ 0)
        target_y = int(h - (lookahead_distance - 0.5) / (1.8 - 0.5) * h)
        target_y = max(0, min(h - 1, target_y))

        # target_y에 대응하는 슬라이딩 윈도우 인덱스 추출
        window_h = h // NUM_WINDOWS
        target_idx = int((h - target_y) / window_h)
        target_idx = max(0, min(NUM_WINDOWS - 1, target_idx))

        # 해당 윈도우의 차선 중심 x좌표 선택
        target_x = center_pts[target_idx][0]

        # 6. 이미지 중앙 대비 가로 픽셀 편차를 물리 거리(m)로 변환
        offset = target_x - (w // 2)
        e_y = offset * self.bev_pixel_to_meter

        # 차선 유효 검출 여부 판단 (이진 마스크 픽셀 총합 기준)
        if np.sum(combined_mask) < 200:
            self.no_lane_count += 1
        else:
            self.no_lane_count = 0

        # 7. 디버그 오버레이 시각화 생성
        # 원본 이미지에 초록색 BEV ROI 영역 다각형 그리기
        debug_orig = image.copy()
        src_draw = self.src_pts.astype(np.int32)
        cv2.polylines(debug_orig, [src_draw], isClosed=True, color=(0, 255, 0), thickness=2)

        # BEV 디버그 이미지 상에 목표값 시각화
        # Lookahead 수평선 (주황색)
        cv2.line(debug_bev, (0, target_y), (w, target_y), (0, 165, 255), 1, cv2.LINE_AA)
        # 타겟 목표점 (노란색 원)
        cv2.circle(debug_bev, (target_x, target_y), 8, (0, 255, 255), -1)
        # 차량 진행 방향 벡터 화살표 (보라색)
        cv2.arrowedLine(debug_bev, (w // 2, h - 1), (target_x, target_y), (255, 0, 255), 3, tipLength=0.1)

        # 이진 마스크 컬러 채널 복제
        mask_bgr = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)

        # 3분할 화면 수평 결합
        combined = np.hstack((debug_orig, debug_bev, mask_bgr))

        # 라벨 및 상태 텍스트 출력
        cv2.putText(combined, "1. Front Camera (ROI)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, f"2. BEV (L_d: {lookahead_distance:.2f}m | e_y: {e_y:+.3f}m)", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, "3. Filtered Lane Mask", (w * 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 리사이즈
        display_w = int(w * 3 * 0.7)
        display_h = int(h * 0.7)
        debug_img = cv2.resize(combined, (display_w, display_h))

        return e_y, debug_img

    def _filter_white(self, hsv):
        """HSV 색공간에서 흰색 차선 후보 마스크를 반환합니다."""
        lower = np.array([WHITE_H_MIN, WHITE_S_MIN, WHITE_V_MIN])
        upper = np.array([WHITE_H_MAX, WHITE_S_MAX, WHITE_V_MAX])
        return cv2.inRange(hsv, lower, upper)

    def _filter_yellow(self, hsv):
        """HSV 색공간에서 노란색 차선 후보 마스크를 반환합니다."""
        lower = np.array([YELLOW_H_MIN, YELLOW_S_MIN, YELLOW_V_MIN])
        upper = np.array([YELLOW_H_MAX, YELLOW_S_MAX, YELLOW_V_MAX])
        return cv2.inRange(hsv, lower, upper)

    def _sliding_window_bev(self, white_mask, yellow_mask, combined_mask, debug_img):
        """BEV 상에서 슬라이딩 윈도우 기법을 적용하여 좌우 차선을 검출하고 추적합니다."""
        h, w = combined_mask.shape
        
        # 노란색 및 흰색 차선 픽셀 총량 확인
        num_yellow = np.sum(yellow_mask > 0)
        num_white = np.sum(white_mask > 0)
        
        # 색상 기반 추적 활성화 여부
        use_color_tracking = (num_yellow > 50) or (num_white > 50)
        
        if use_color_tracking:
            left_mask = yellow_mask
            right_mask = white_mask
            
            left_base = self.prev_left_x
            right_base = self.prev_right_x
            
            if num_yellow > 50:
                left_hist = np.sum(left_mask[h // 2:, :], axis=0)
                if np.max(left_hist) > 10:
                    left_base = np.argmax(left_hist)
                        
            if num_white > 50:
                right_hist = np.sum(right_mask[h // 2:, :], axis=0)
                if np.max(right_hist) > 10:
                    right_base = np.argmax(right_hist)
        else:
            # 올화이트 차선 또는 노란색/흰색 모두 검출 안됨
            left_mask = combined_mask
            right_mask = combined_mask
            
            histogram = np.sum(combined_mask[h // 2:, :], axis=0)
            midpoint = w // 2
            
            left_base = self.prev_left_x
            right_base = self.prev_right_x
            
            if np.max(histogram[:midpoint]) > 100:
                hist_left = np.argmax(histogram[:midpoint])
                if abs(hist_left - self.prev_left_x) < 120 or self.prev_left_x == int(w * 0.2):
                    left_base = hist_left
                    
            if np.max(histogram[midpoint:]) > 100:
                hist_right = np.argmax(histogram[midpoint:]) + midpoint
                if abs(hist_right - self.prev_right_x) < 120 or self.prev_right_x == int(w * 0.8):
                    right_base = hist_right

        left_current = left_base
        right_current = right_base

        left_pts = []
        right_pts = []
        center_pts = []

        window_h = h // NUM_WINDOWS
        lane_width_bev = self.lane_width_bev

        for i in range(NUM_WINDOWS):
            y_low = h - (i + 1) * window_h
            y_high = h - i * window_h
            y_center = (y_low + y_high) // 2

            # 음수 슬라이싱 방지를 위한 클리핑 적용 (안전한 인덱스 바운딩)
            win_xl_low = max(0, min(w, left_current - WINDOW_MARGIN))
            win_xl_high = max(0, min(w, left_current + WINDOW_MARGIN))
            win_xr_low = max(0, min(w, right_current - WINDOW_MARGIN))
            win_xr_high = max(0, min(w, right_current + WINDOW_MARGIN))

            if debug_img is not None:
                # 윈도우 사각형 표시 (좌: 녹색, 우: 적색)
                cv2.rectangle(debug_img, (win_xl_low, y_low), (win_xl_high, y_high), (0, 255, 0), 1)
                cv2.rectangle(debug_img, (win_xr_low, y_low), (win_xr_high, y_high), (0, 0, 255), 1)

            left_area = left_mask[y_low:y_high, win_xl_low:win_xl_high]
            right_area = right_mask[y_low:y_high, win_xr_low:win_xr_high]
            
            left_detected = np.sum(left_area) > 100
            right_detected = np.sum(right_area) > 100

            if left_detected and right_detected:
                cand_left = int(win_xl_low + np.mean(np.where(left_area > 0)[1]))
                cand_right = int(win_xr_low + np.mean(np.where(right_area > 0)[1]))
                
                # 검출된 차선폭 검증
                measured_width = cand_right - cand_left
                # 예상 차선폭 범위 내인 경우에만 둘 다 인정
                if abs(measured_width - lane_width_bev) < 60:
                    left_current = cand_left
                    right_current = cand_right
                    # 차선폭 동적 업데이트
                    self.lane_width_bev = int(0.9 * self.lane_width_bev + 0.1 * measured_width)
                    lane_width_bev = self.lane_width_bev
                else:
                    # 차선폭이 비정상인 경우, 이전 윈도우(바로 아래 윈도우) 위치로부터의 변화량이 적은 쪽을 신뢰
                    prev_l = left_pts[-1][0] if len(left_pts) > 0 else left_base
                    prev_r = right_pts[-1][0] if len(right_pts) > 0 else right_base
                    
                    shift_l = abs(cand_left - prev_l)
                    shift_r = abs(cand_right - prev_r)
                    
                    if shift_l <= shift_r:
                        left_current = cand_left
                        # 우측 차선은 학습된 가로폭으로 복원
                        right_current = left_current + lane_width_bev
                    else:
                        right_current = cand_right
                        # 좌측 차선은 학습된 가로폭으로 복원
                        left_current = right_current - lane_width_bev
            elif left_detected:
                left_current = int(win_xl_low + np.mean(np.where(left_area > 0)[1]))
                # 우측 차선은 학습된 가로폭으로 복원
                right_current = left_current + lane_width_bev
            elif right_detected:
                right_current = int(win_xr_low + np.mean(np.where(right_area > 0)[1]))
                # 좌측 차선은 학습된 가로폭으로 복원
                left_current = right_current - lane_width_bev
            # 둘 다 감지 안 되면 이전 윈도우 값을 전파

            left_pts.append((left_current, y_center))
            right_pts.append((right_current, y_center))
            
            center_x = (left_current + right_current) // 2
            center_pts.append((center_x, y_center))

            if debug_img is not None:
                cv2.circle(debug_img, (left_current, y_center), 3, (255, 0, 0), -1)
                cv2.circle(debug_img, (right_current, y_center), 3, (0, 0, 255), -1)
                cv2.circle(debug_img, (center_x, y_center), 3, (0, 255, 0), -1)

        # 차선 정보 캐싱 (다음 프레임 필터링 적용)
        self.prev_left_x = int(0.75 * self.prev_left_x + 0.25 * left_current)
        self.prev_right_x = int(0.75 * self.prev_right_x + 0.25 * right_current)
        self.prev_center_x = (self.prev_left_x + self.prev_right_x) // 2

        return left_pts, right_pts, center_pts, debug_img
