#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
도로 표시 인식 모듈 (Road Sign Detector)
- 도로 노면에 적힌 "어린이 보호구역" 및 "보호구역 해제" 문구를 인식합니다.
- OCR 대신 노란색 문자 영역의 패턴 매칭으로 판별합니다.
- 카메라 영상의 도로 바닥면을 분석합니다.
"""
import cv2
import numpy as np
# ============================================
# 상수 정의
# ============================================
# 도로 노면 문구 인식 ROI (도로 바닥 영역)
ROAD_SIGN_TOP_RATIO = 0.55     # 도로 하단 영역 시작
ROAD_SIGN_BOTTOM_RATIO = 0.90  # 도로 하단 영역 끝
# 노란색 문구 HSV 임계값 (도로에 쓰인 노란 글씨)
SIGN_YELLOW_H_MIN, SIGN_YELLOW_H_MAX = 15, 40
SIGN_YELLOW_S_MIN, SIGN_YELLOW_S_MAX = 80, 255
SIGN_YELLOW_V_MIN, SIGN_YELLOW_V_MAX = 150, 255
# 문구 면적 임계값
SIGN_MIN_AREA = 1000       # 최소 면적 (노이즈 제거)
SIGN_MAX_AREA = 50000      # 최대 면적
# 어린이 보호구역 판별 기준
# "어린이 보호구역"은 노란색 글자가 넓게 분포
SCHOOL_ZONE_MIN_AREA = 2000
# "해제"는 상대적으로 작은 영역
RELEASE_MIN_AREA = 800
# 도로 상태 열거
ZONE_NORMAL = "NORMAL"             # 일반 구간
ZONE_SCHOOL = "SCHOOL_ZONE"       # 어린이 보호구역
ZONE_RELEASE = "ZONE_RELEASE"     # 보호구역 해제
class RoadSignDetector:
    """
    도로 노면에 적힌 문구(어린이 보호구역, 해제)를 인식하여
    속도 제한 정보를 제공합니다.
    """
    def __init__(self):
        # 현재 구역 상태
        self.current_zone = ZONE_NORMAL
        # 연속 감지 카운터 (안정적 판단용)
        self.school_zone_count = 0
        self.release_count = 0
        self.confirm_threshold = 5  # 5프레임 연속 감지 시 확정
        # 이전 프레임의 노란색 영역 면적 (변화 감지용)
        self.prev_yellow_area = 0
    # ============================================
    # 메인 감지 함수
    # ============================================
    def detect(self, image):
        """
        카메라 영상에서 도로 노면 문구를 인식합니다.
        Args:
            image: BGR 형식 카메라 이미지
        Returns:
            zone_state (str): 현재 구역 상태
            speed_limit (float): 권장 속도 제한 비율 (0.0~1.0, 1.0이면 제한 없음)
        """
        if image is None:
            return self.current_zone, 1.0
        h, w = image.shape[:2]
        # ROI 추출: 도로 바닥면
        roi_top = int(h * ROAD_SIGN_TOP_RATIO)
        roi_bottom = int(h * ROAD_SIGN_BOTTOM_RATIO)
        roi = image[roi_top:roi_bottom, :]
        # HSV 변환 및 노란색 문구 필터링
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(
            hsv,
            np.array([SIGN_YELLOW_H_MIN, SIGN_YELLOW_S_MIN, SIGN_YELLOW_V_MIN]),
            np.array([SIGN_YELLOW_H_MAX, SIGN_YELLOW_S_MAX, SIGN_YELLOW_V_MAX])
        )
        # 모폴로지 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        # 노란색 영역 면적 계산
        yellow_area = cv2.countNonZero(yellow_mask)
        # 컨투어 분석
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        # 큰 노란색 영역이 있는지 확인
        large_yellow_regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > SIGN_MIN_AREA:
                large_yellow_regions.append((cnt, area))
        # 문구 판별 로직
        total_sign_area = sum(a for _, a in large_yellow_regions)
        if total_sign_area > SCHOOL_ZONE_MIN_AREA:
            # 노란색 문구가 넓게 분포 → "어린이 보호구역" 또는 "해제" 판별
            # 문자 영역의 분포 패턴으로 판별
            if len(large_yellow_regions) > 0:
                # 가장 큰 영역의 바운딩 박스
                all_points = np.vstack([cnt for cnt, _ in large_yellow_regions])
                x, y, bw, bh = cv2.boundingRect(all_points)
                # "어린이 보호구역"은 글자가 많아 넓은 영역
                # "보호구역 해제"/"해제"는 글자가 적어 좁은 영역
                if total_sign_area > SCHOOL_ZONE_MIN_AREA * 2 and bw > w * 0.3:
                    # 넓은 영역 → "어린이 보호구역"
                    self.school_zone_count += 2
                    self.release_count = max(0, self.release_count - 1)
                elif total_sign_area > RELEASE_MIN_AREA and self.current_zone == ZONE_SCHOOL:
                    # 현재 보호구역 상태에서 다시 노란 문구 → "해제"일 가능성
                    self.release_count += 2
                    self.school_zone_count = max(0, self.school_zone_count - 1)
                else:
                    self.school_zone_count += 1
        else:
            # 노란색 문구가 거의 없음
            self.school_zone_count = max(0, self.school_zone_count - 1)
            self.release_count = max(0, self.release_count - 1)
        # 상태 전이 판단
        if self.current_zone == ZONE_NORMAL:
            if self.school_zone_count >= self.confirm_threshold:
                self.current_zone = ZONE_SCHOOL
                self.school_zone_count = 0
                self.release_count = 0
        elif self.current_zone == ZONE_SCHOOL:
            if self.release_count >= self.confirm_threshold:
                self.current_zone = ZONE_NORMAL
                self.school_zone_count = 0
                self.release_count = 0
        # 속도 제한 비율 반환
        speed_limit = 0.4 if self.current_zone == ZONE_SCHOOL else 1.0
        self.prev_yellow_area = yellow_area
        return self.current_zone, speed_limit
    # ============================================
    # 구역 상태 초기화
    # ============================================
    def reset(self):
        """구역 상태를 초기화합니다."""
        self.current_zone = ZONE_NORMAL
        self.school_zone_count = 0
        self.release_count = 0
