#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
차선 인식 모듈 (Lane Detector)
- 카메라 영상에서 차선(흰색 실선, 노란색 점선/실선)을 검출하여
  차량이 차선 중앙을 유지하도록 조향각(steering offset)을 계산합니다.
- HSV 색공간 기반 필터링과 슬라이딩 윈도우 기법을 사용합니다.
"""
import cv2
import numpy as np
# ============================================
# 상수 정의
# ============================================
# 카메라 이미지 크기 (시뮬레이터 기본값)
IMG_WIDTH = 640
IMG_HEIGHT = 480
# ROI(관심 영역) 설정 - 이미지 하단부만 사용
ROI_TOP_RATIO = 0.55       # 상단 55% 이상은 무시
ROI_BOTTOM_RATIO = 0.95    # 하단 5%도 차체가 보이므로 제외
# 슬라이딩 윈도우 설정
NUM_WINDOWS = 9            # 세로 방향 윈도우 개수
WINDOW_MARGIN = 60         # 윈도우 좌우 탐색 범위 (픽셀)
MIN_PIX_RECENTER = 30      # 윈도우 재조정 최소 픽셀 수
# HSV 색공간 임계값 - 흰색 차선
WHITE_H_MIN, WHITE_H_MAX = 0, 180
WHITE_S_MIN, WHITE_S_MAX = 0, 40
WHITE_V_MIN, WHITE_V_MAX = 200, 255
# HSV 색공간 임계값 - 노란색 차선
YELLOW_H_MIN, YELLOW_H_MAX = 15, 35
YELLOW_S_MIN, YELLOW_S_MAX = 80, 255
YELLOW_V_MIN, YELLOW_V_MAX = 150, 255
# 차선 중심 오프셋 → 조향각 변환 계수
STEER_GAIN = 0.4
class LaneDetector:
    """
    카메라 영상에서 좌/우 차선을 감지하고,
    차량이 차선 중앙을 유지하기 위한 조향 오프셋을 반환합니다.
    """
    def __init__(self):
        # 이전 프레임의 차선 중심 위치 저장 (차선이 한쪽만 보일 때 보정용)
        self.prev_left_x = IMG_WIDTH // 4
        self.prev_right_x = IMG_WIDTH * 3 // 4
        self.prev_center_x = IMG_WIDTH // 2
        # 차선 검출 실패 카운터
        self.no_lane_count = 0
    # ============================================
    # 메인 처리 함수: 이미지를 입력받아 조향 오프셋 반환
    # ============================================
    def detect(self, image):
        """
        카메라 영상에서 차선을 검출하고 조향각 오프셋을 계산합니다.
        Args:
            image: BGR 형식의 카메라 이미지 (numpy array)
        Returns:
            steer_offset (float): 조향 오프셋 (-값: 왼쪽, +값: 오른쪽)
            debug_image (numpy array): 디버그용 시각화 이미지
        """
        if image is None:
            return 0.0, None
        h, w = image.shape[:2]
        # 1단계: ROI 추출
        roi_top = int(h * ROI_TOP_RATIO)
        roi_bottom = int(h * ROI_BOTTOM_RATIO)
        roi = image[roi_top:roi_bottom, :]
        # 2단계: HSV 변환 및 차선 색상 필터링
        white_mask = self._filter_white(roi)
        yellow_mask = self._filter_yellow(roi)
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        # 3단계: 도로(검은색/어두운 회색) 영역 검출 및 차선 마스킹
        # 도로 이외의 나무, 잔디밭에서 올라오는 노이즈 차단을 위해 도로 인접성 검사
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_road = np.array([0, 0, 30])
        upper_road = np.array([180, 50, 160])
        road_mask = cv2.inRange(hsv, lower_road, upper_road)
        
        # 도로 마스크를 25x25 크기로 팽창(Dilate)시켜 도로 양 가장자리의 차선 영역까지 덮게 함
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        dilated_road_mask = cv2.dilate(road_mask, kernel_dilate, iterations=1)
        
        # 도로 인접 영역 내부에서 발견된 흰색/노란색 필터만 최종 차선 후보로 승인
        combined_mask = cv2.bitwise_and(combined_mask, dilated_road_mask)

        # 4단계: 노이즈 제거 (모폴로지 연산)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        # 5단계: 슬라이딩 윈도우로 좌/우 차선 픽셀 탐색
        left_x, right_x, debug_img = self._sliding_window(combined_mask, roi.copy())
        # 6단계: 차선 중심 계산 및 조향 오프셋 산출
        steer_offset = self._compute_steering(left_x, right_x, w)
        return steer_offset, debug_img
    # ============================================
    # 흰색 차선 필터링 함수
    # ============================================
    def _filter_white(self, roi):
        """HSV 색공간에서 흰색 영역을 마스크로 추출합니다."""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array([WHITE_H_MIN, WHITE_S_MIN, WHITE_V_MIN])
        upper = np.array([WHITE_H_MAX, WHITE_S_MAX, WHITE_V_MAX])
        return cv2.inRange(hsv, lower, upper)
    # ============================================
    # 노란색 차선 필터링 함수
    # ============================================
    def _filter_yellow(self, roi):
        """HSV 색공간에서 노란색 영역을 마스크로 추출합니다."""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array([YELLOW_H_MIN, YELLOW_S_MIN, YELLOW_V_MIN])
        upper = np.array([YELLOW_H_MAX, YELLOW_S_MAX, YELLOW_V_MAX])
        return cv2.inRange(hsv, lower, upper)
    # ============================================
    # 슬라이딩 윈도우 기반 차선 탐색 함수
    # ============================================
    def _sliding_window(self, binary_mask, debug_img):
        """
        이진 마스크에서 슬라이딩 윈도우를 사용하여
        좌/우 차선의 x좌표 평균을 구합니다.
        """
        h, w = binary_mask.shape
        # 히스토그램으로 초기 차선 위치 추정 (하단 절반 사용)
        histogram = np.sum(binary_mask[h // 2:, :], axis=0)
        midpoint = w // 2
        left_base = np.argmax(histogram[:midpoint]) if np.max(histogram[:midpoint]) > 0 else self.prev_left_x
        right_base = np.argmax(histogram[midpoint:]) + midpoint if np.max(histogram[midpoint:]) > 0 else self.prev_right_x
        # 윈도우 높이
        window_h = h // NUM_WINDOWS
        # 0이 아닌 픽셀 좌표 추출
        nonzero = binary_mask.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
        # 현재 윈도우 중심점
        left_current = left_base
        right_current = right_base
        # 좌/우 차선 픽셀 인덱스 수집
        left_lane_inds = []
        right_lane_inds = []
        for win_idx in range(NUM_WINDOWS):
            # 윈도우 영역 계산 (아래에서 위로)
            win_y_low = h - (win_idx + 1) * window_h
            win_y_high = h - win_idx * window_h
            # 왼쪽 윈도우
            win_xl_low = max(0, left_current - WINDOW_MARGIN)
            win_xl_high = min(w, left_current + WINDOW_MARGIN)
            # 오른쪽 윈도우
            win_xr_low = max(0, right_current - WINDOW_MARGIN)
            win_xr_high = min(w, right_current + WINDOW_MARGIN)
            # 디버그 시각화: 윈도우 사각형 표시
            if debug_img is not None:
                cv2.rectangle(debug_img, (win_xl_low, win_y_low),
                              (win_xl_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(debug_img, (win_xr_low, win_y_low),
                              (win_xr_high, win_y_high), (0, 0, 255), 2)
            # 윈도우 내 픽셀 인덱스 수집
            good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                         (nonzero_x >= win_xl_low) & (nonzero_x < win_xl_high)).nonzero()[0]
            good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                          (nonzero_x >= win_xr_low) & (nonzero_x < win_xr_high)).nonzero()[0]
            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)
            # 충분한 픽셀이 있으면 윈도우 중심 재조정
            if len(good_left) > MIN_PIX_RECENTER:
                left_current = int(np.mean(nonzero_x[good_left]))
            if len(good_right) > MIN_PIX_RECENTER:
                right_current = int(np.mean(nonzero_x[good_right]))
        # 모든 윈도우의 인덱스 합치기
        left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([])
        right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([])
        # 최종 좌/우 차선 중심 x좌표 계산
        left_x = int(np.mean(nonzero_x[left_lane_inds])) if len(left_lane_inds) > 50 else None
        right_x = int(np.mean(nonzero_x[right_lane_inds])) if len(right_lane_inds) > 50 else None
        return left_x, right_x, debug_img
    # ============================================
    # 조향 오프셋 계산 함수
    # ============================================
    def _compute_steering(self, left_x, right_x, img_width):
        """
        좌/우 차선 위치로부터 차량 중앙과의 오프셋을 계산하고
        조향각 오프셋으로 변환합니다.
        """
        center = img_width // 2
        lane_width_estimate = img_width // 2  # 차선 폭 추정값
        if left_x is not None and right_x is not None:
            # 양쪽 차선 모두 감지됨 → 정확한 중심 계산
            lane_center = (left_x + right_x) // 2
            self.prev_left_x = left_x
            self.prev_right_x = right_x
            self.no_lane_count = 0
        elif left_x is not None:
            # 왼쪽 차선만 감지됨 → 추정 폭으로 우측 차선 추정
            lane_center = left_x + lane_width_estimate // 2
            self.prev_left_x = left_x
            self.no_lane_count = 0
        elif right_x is not None:
            # 오른쪽 차선만 감지됨 → 추정 폭으로 좌측 차선 추정
            lane_center = right_x - lane_width_estimate // 2
            self.prev_right_x = right_x
            self.no_lane_count = 0
        else:
            # 양쪽 모두 감지 실패 → 이전 값 사용
            lane_center = self.prev_center_x
            self.no_lane_count += 1
        self.prev_center_x = lane_center
        # 이미지 중앙과 차선 중심의 차이 → 조향 오프셋
        offset = lane_center - center
        steer_offset = offset * STEER_GAIN
        return steer_offset
