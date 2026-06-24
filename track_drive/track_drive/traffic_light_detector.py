#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
신호등 인식 모듈 (Traffic Light Detector)
"""
import cv2
import numpy as np

# ROI
TL_ROI_TOP = 0
TL_ROI_BOTTOM_RATIO = 0.35   # 좀 더 위쪽으로 제한하여 배경 제외
TL_ROI_LEFT_RATIO = 0.1
TL_ROI_RIGHT_RATIO = 0.9

RED_H_LOW1, RED_H_HIGH1 = 0, 10
RED_H_LOW2, RED_H_HIGH2 = 160, 180
RED_S_MIN, RED_V_MIN = 80, 150

YEL_H_LOW, YEL_H_HIGH = 15, 35
YEL_S_MIN, YEL_V_MIN = 80, 150

GRN_H_LOW, GRN_H_HIGH = 40, 90
GRN_S_MIN, GRN_V_MIN = 80, 150

MIN_AREA_THRESHOLD = 15

SIGNAL_UNKNOWN = "UNKNOWN"
SIGNAL_RED = "RED"
SIGNAL_YELLOW = "YELLOW"
SIGNAL_GREEN = "GREEN"
SIGNAL_LEFT_ARROW = "LEFT_ARROW"

class TrafficLightDetector:
    def __init__(self):
        self.signal_counts = {
            SIGNAL_RED: 0,
            SIGNAL_YELLOW: 0,
            SIGNAL_GREEN: 0,
            SIGNAL_LEFT_ARROW: 0,
        }
        self.current_signal = SIGNAL_UNKNOWN
        self.confirm_threshold = 3

    def detect(self, image):
        if image is None:
            return self.current_signal, None

        h, w = image.shape[:2]
        roi_top = TL_ROI_TOP
        roi_bottom = int(h * TL_ROI_BOTTOM_RATIO)
        roi_left = int(w * TL_ROI_LEFT_RATIO)
        roi_right = int(w * TL_ROI_RIGHT_RATIO)
        roi = image[roi_top:roi_bottom, roi_left:roi_right]

        debug_img = roi.copy()
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        red_area = self._detect_red(hsv)
        yellow_area = self._detect_yellow(hsv)
        green_area = self._detect_green(hsv)
        left_arrow_detected = self._detect_left_arrow(hsv, roi)

        frame_signal = SIGNAL_UNKNOWN

        if left_arrow_detected:
            frame_signal = SIGNAL_LEFT_ARROW
        elif green_area > MIN_AREA_THRESHOLD and green_area >= red_area and green_area >= yellow_area:
            frame_signal = SIGNAL_GREEN
        elif red_area > MIN_AREA_THRESHOLD and red_area >= yellow_area:
            frame_signal = SIGNAL_RED
        elif yellow_area > MIN_AREA_THRESHOLD:
            frame_signal = SIGNAL_YELLOW

        if frame_signal != SIGNAL_UNKNOWN:
            for key in self.signal_counts:
                if key == frame_signal:
                    self.signal_counts[key] += 1
                else:
                    self.signal_counts[key] = max(0, self.signal_counts[key] - 1)

            if self.signal_counts[frame_signal] >= self.confirm_threshold:
                self.current_signal = frame_signal
        else:
            # 빛이 안 보이면 점진적으로 카운트 감소
            for key in self.signal_counts:
                self.signal_counts[key] = max(0, self.signal_counts[key] - 1)

        if debug_img is not None:
            cv2.putText(debug_img, f"Signal: {self.current_signal}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(debug_img, f"R:{int(red_area)} Y:{int(yellow_area)} G:{int(green_area)}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return self.current_signal, debug_img

    def _get_max_circular_area(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA_THRESHOLD:
                continue
            x, y, w, h_box = cv2.boundingRect(cnt)
            aspect_ratio = w / max(h_box, 1)
            # 신호등 불빛은 거의 원형이지만 번짐 효과 고려하여 범위 확대
            if 0.2 <= aspect_ratio <= 5.0:
                if area > max_area:
                    max_area = area
        return max_area

    def _detect_red(self, hsv):
        mask1 = cv2.inRange(hsv, np.array([RED_H_LOW1, RED_S_MIN, RED_V_MIN]), np.array([RED_H_HIGH1, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([RED_H_LOW2, RED_S_MIN, RED_V_MIN]), np.array([RED_H_HIGH2, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)
        return self._get_max_circular_area(mask)

    def _detect_yellow(self, hsv):
        mask = cv2.inRange(hsv, np.array([YEL_H_LOW, YEL_S_MIN, YEL_V_MIN]), np.array([YEL_H_HIGH, 255, 255]))
        return self._get_max_circular_area(mask)

    def _detect_green(self, hsv):
        mask = cv2.inRange(hsv, np.array([GRN_H_LOW, GRN_S_MIN, GRN_V_MIN]), np.array([GRN_H_HIGH, 255, 255]))
        return self._get_max_circular_area(mask)

    def _detect_left_arrow(self, hsv, roi_bgr):
        green_mask = cv2.inRange(hsv, np.array([GRN_H_LOW, GRN_S_MIN, GRN_V_MIN]), np.array([GRN_H_HIGH, 255, 255]))
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA_THRESHOLD:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = bw / max(bh, 1)
            # 화살표는 가로로 긴 형태
            if aspect_ratio > 1.2 and area > MIN_AREA_THRESHOLD:
                return True
        return False

    def reset(self):
        for key in self.signal_counts:
            self.signal_counts[key] = 0
        self.current_signal = SIGNAL_UNKNOWN
