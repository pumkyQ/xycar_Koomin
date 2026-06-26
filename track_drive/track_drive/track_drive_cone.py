#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
import time
import cv2
import numpy as np
import math
from rclpy.node import Node
from xycar_msgs.msg import XycarMotor
from sensor_msgs.msg import Image, LaserScan
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge

# 자체 모듈 임포트
from track_drive.lane_detector import LaneDetector
from track_drive.lidar_drive import ConeDriver
from track_drive.obstacle_detector import ObstacleDetector
from track_drive.road_sign_detector import RoadSignDetector, ZONE_SCHOOL
from track_drive.traffic_light_detector import (
    TrafficLightDetector, SIGNAL_GREEN, SIGNAL_RED,
    SIGNAL_YELLOW, SIGNAL_LEFT_ARROW, SIGNAL_UNKNOWN
)

#====================================================================
# [Drive States]
# 제주도 대회 FSM 설계 패턴을 벤치마킹한 주행 상태 정의
#====================================================================
class DriveState:
    WAIT_FOR_GREEN = 'wait_for_green'     # 1. 신호 대기 상태
    CONE_DRIVING = 'cone_driving'         # 2. 라바콘(문코스) 회피 주행 상태
    LANE_DRIVING = 'lane_driving'         # 3. Pure Pursuit 차선 주행 상태
    PEDESTRIAN_STOP = 'pedestrian_stop'   # 4. 보행자 감지 긴급 정지 상태
    FINISHED = 'finished'                 # 3바퀴 완주 후 정지 상태

class TrackDriverNode(Node):
    def __init__(self):
        super().__init__('driver')
        self.get_logger().info('===== 국민대 대회 예선과제 1번 라바콘 통합 제어 노드 시작 =====')

        # 센서 데이터 및 브릿지 초기화
        self.image = None
        self.lidar_ranges = None
        self.bridge = CvBridge()
        self.motor_msg = XycarMotor()

        # 인지 모듈 객체 초기화
        self.lane_detector = LaneDetector()
        self.traffic_detector = TrafficLightDetector()
        self.obstacle_detector = ObstacleDetector()
        self.cone_driver = ConeDriver(target_speed=20.0, kp=70.0) # 친구의 기본 고속 설정(20.0) 적용

        # 주행 상태 머신 및 타이밍 초기화
        self.current_drive_state = DriveState.WAIT_FOR_GREEN
        self.lap_count = 0
        self.total_laps = 3

        # 퓨어퍼슛 파라미터 설정 (LANE_DRIVING 전용)
        self.wheelbase = 0.33
        self.focal_length = 350.0
        self.steer_gain = 4.0       # 최근 피드백 조향 감도(4.0) 적용
        self.is_curve_mode = False  # 직선/급커브 판정을 위한 상태 변수
        self.curve_enter_threshold = 25.0 # 급커브 모드 진입 임계값
        self.curve_exit_threshold = 16.0  # 급커브 모드 탈출 임계값 (원래 안정적인 값으로 복구)
        self.last_curve_time = 0.0        # 마지막으로 곡선이 감지된 시점
        self.curve_exit_delay = 0.8       # 곡선 탈출 지연 시간 (0.8초로 복구)
        self.filtered_curvature = 0.0     # 필터링된 곡률 상태값
        self.curv_alpha = 0.25            # EMA 필터 계수 (반응성을 위해 0.10에서 0.25로 상향 복구)
        self.lookahead_min = 0.5    # 최소 전방주시거리
        self.lookahead_max = 1.5    # 최대 전방주시거리 (1.4에서 1.5로 복구)
        
        # 속도 기본값 설정 (LANE_DRIVING 전용)
        self.speed_max = 10.0      # 직진 최고 속도 (안정적인 10.0으로 복구)
        self.speed_min = 4.0       # 커브 최저 속도 (안정적인 4.0으로 복구)
        self.speed_stop = 0.0       # 정지 속도
        
        # 보행자 안전 대기 관련 변수
        self.pedestrian_clear_time = 0.0
        self.recovery_delay_sec = 3.0  # 보행자 이탈 후 재출발까지 안전 대기 시간 (3초)
        self.prev_state_before_ped = DriveState.LANE_DRIVING

        # 라바콘(Phase 2) 타이머 변수
        self.phase2_start_time = 0.0

        # 바퀴 수 완주 측정을 위한 타이밍 변수
        self.start_time = time.time()
        self.last_lap_time = time.time()

        # 조향 변화율 제한 및 필터용 변수 (LANE_DRIVING 전용)
        self.prev_steer = 0.0
        self.prev_steer_error = 0.0
        self.steer_integral = 0.0

        # 콘 개수 실시간 디버그용 상태 변수
        self.detected_cones_count = 0

        # ROS2 퍼블리셔 및 서브스크라이버 설정
        self.motor_pub = self.create_publisher(XycarMotor, 'xycar_motor', 10)
        self.sub_front = self.create_subscription(
            Image, '/usb_cam/image_raw/front',
            self.cam_callback, qos_profile_sensor_data)
        self.sub_lidar = self.create_subscription(
            LaserScan, '/scan',
            self.lidar_callback, qos_profile_sensor_data)

    def cam_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def lidar_callback(self, msg):
        self.lidar_ranges = msg.ranges

    def drive(self, angle, speed):
        """
        차량 제어 토픽(xycar_motor)을 발행합니다.
        """
        clamped_angle = max(-100.0, min(100.0, angle))
        
        self.motor_msg.angle = float(clamped_angle)
        self.motor_msg.speed = float(speed)
        self.motor_pub.publish(self.motor_msg)

    #====================================================================
    # [Steering Controller]
    # Pure Pursuit 기하학적 제어기
    #====================================================================
    def pure_pursuit_steering(self, e_y, lookahead_distance):
        """
        Pure Pursuit 알고리즘을 사용하여 조향각을 계산합니다.
        """
        delta = math.atan2(2.0 * self.wheelbase * e_y, lookahead_distance ** 2)
        angle_deg = math.degrees(delta)
        
        # 조향 게인 반영 및 출력 범위 제한 (-100 ~ 100)
        steer_cmd = angle_deg * self.steer_gain
        return max(-100.0, min(100.0, steer_cmd))

    def _map_speed_by_steer(self, steer_cmd):
        """
        조향각의 절대값에 따라 차량 속도를 동적으로 매핑합니다.
        """
        steer_ratio = min(1.0, abs(steer_cmd) / 100.0)
        steer_ratio_curved = steer_ratio ** 1.5  # 1.5승을 취해 급커브 진입 초기에 더 민감하게 감속
        return self.speed_max - steer_ratio_curved * (self.speed_max - self.speed_min)

    def detect_asphalt(self, cv_image):
        """
        카메라 이미지를 바탕으로 현재 아스팔트 차선 위에 확실하게 진입했는지 판단합니다.
        바닥의 '검은색 아스팔트' 영역을 인식합니다.
        """
        if cv_image is None:
            return False
            
        h, w = cv_image.shape[:2]
        # 차량 바로 앞 바닥(ROI)을 잘라냅니다.
        roi = cv_image[350:480, 200:440]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 검은색/어두운 회색 아스팔트 색상 범위
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 60, 90])
        
        mask = cv2.inRange(hsv, lower_black, upper_black)
        
        black_pixels = cv2.countNonZero(mask)
        total_pixels = roi.shape[0] * roi.shape[1]
        
        # 해당 영역의 50% 이상이 검은색이면 아스팔트 진입으로 판단
        return black_pixels > (total_pixels * 0.5)

    #====================================================================
    # [FSM State Transitions]
    # 계층적 상태 판단 패턴 적용 (라바콘 통합 버전)
    #====================================================================
    def _get_next_state(self):
        """
        현재 차량 상태 및 센서 데이터를 바탕으로 다음 상태를 결정합니다.
        """
        # 1. 완주 종료 상태 고정
        if self.current_drive_state == DriveState.FINISHED:
            return DriveState.FINISHED

        # 2. 신호등 대기 상태 처리
        if self.current_drive_state == DriveState.WAIT_FOR_GREEN:
            signal, _ = self.traffic_detector.detect(self.image)
            if signal == SIGNAL_GREEN:
                self.get_logger().info("★ 녹색 신호 감지! [Phase 2] 라바콘 회피 주행을 시작합니다.")
                self.start_time = time.time()
                self.last_lap_time = time.time()
                self.phase2_start_time = time.time() # 라바콘 시작 시간 마킹
                return DriveState.CONE_DRIVING
            return DriveState.WAIT_FOR_GREEN

        # 3. 완주 체크 (차선 주행 상태에서만 수행)
        if self.current_drive_state == DriveState.LANE_DRIVING:
            self.check_lap_completion()

        # 4. 보행자 감지 판단 (모든 주행 상태에서 긴급 정지 수행 가능)
        ped_info = self.obstacle_detector.detect_pedestrian(self.image, self.lidar_ranges)

        if self.current_drive_state in (DriveState.LANE_DRIVING, DriveState.CONE_DRIVING):
            if ped_info['pedestrian_detected'] and ped_info['should_stop'] and ped_info['pedestrian_direction'] == 'CENTER':
                self.get_logger().warn("⚠ 전방 보행자 감지! 긴급 정지 상태로 전환합니다. -> PEDESTRIAN_STOP")
                self.pedestrian_clear_time = time.time()
                self.prev_state_before_ped = self.current_drive_state # 복귀할 이전 상태 저장
                return DriveState.PEDESTRIAN_STOP

        # 보행자 긴급 정지 상태 제어
        if self.current_drive_state == DriveState.PEDESTRIAN_STOP:
            if ped_info['pedestrian_detected'] and ped_info['should_stop'] and ped_info['pedestrian_direction'] == 'CENTER':
                self.pedestrian_clear_time = time.time()
                return DriveState.PEDESTRIAN_STOP
            else:
                # 보행자가 사라진 상태에서 3초 경과 후 원래 주행 모드로 복귀
                elapsed = time.time() - self.pedestrian_clear_time
                if elapsed < self.recovery_delay_sec:
                    return DriveState.PEDESTRIAN_STOP
                restore_state = self.prev_state_before_ped
                self.get_logger().info(f"★ 보행자 이탈 및 안전 대기 시간({self.recovery_delay_sec}초) 경과 완료 ➔ {restore_state} 복귀")
                return restore_state

        # 5. 라바콘 코스 돌파 및 아스팔트 차선 진입 조건 판단
        if self.current_drive_state == DriveState.CONE_DRIVING:
            elapsed = time.time() - self.phase2_start_time
            # 출발 직후 그리드의 검은 바닥 등을 아스팔트로 잘못 인식하지 않도록 3초의 쿨다운 부여
            if elapsed > 3.0:
                if self.detect_asphalt(self.image):
                    self.get_logger().info("★ 아스팔트 차선 진입 감지! [Phase 3] 차선 주행 모드로 전환합니다.")
                    return DriveState.LANE_DRIVING
            return DriveState.CONE_DRIVING

        return DriveState.LANE_DRIVING

    def check_lap_completion(self):
        """차량이 트랙 한 바퀴를 완주하여 출발 지점의 신호등을 다시 감지할 때를 판별합니다."""
        if self.current_drive_state in (DriveState.WAIT_FOR_GREEN, DriveState.FINISHED):
            return

        signal, _ = self.traffic_detector.detect(self.image)
        
        if signal != SIGNAL_UNKNOWN:
            current_time = time.time()
            if current_time - self.last_lap_time > 25.0:
                self.lap_count += 1
                self.last_lap_time = current_time
                self.get_logger().info(f"★★★ Lap {self.lap_count} 완료! (소요 시간: {current_time - self.start_time:.1f}초) ★★★")

    def _maybe_log_status(self, steer_cmd, speed_cmd, raw_curvature=0.0):
        """
        차량의 현재 주행 상태와 제어 토픽 값을 디버그 로깅합니다 (5Hz로 스로틀링).
        """
        if not hasattr(self, '_log_counter'):
            self._log_counter = 0
        self._log_counter += 1
        if self._log_counter % 10 != 0:
            return

        self.get_logger().info(
            f"[FSM STATUS] State: {self.current_drive_state} | "
            f"CurveMode: {self.is_curve_mode} | "
            f"Curv(F/R): {self.filtered_curvature:.1f}/{raw_curvature:.1f} | "
            f"Steer: {steer_cmd:.1f} | Speed: {speed_cmd:.1f} | "
            f"Laps: {self.lap_count}/{self.total_laps}"
        )

    def main_loop(self):
        # 카메라 데이터 수신 시까지 대기
        while rclpy.ok() and self.image is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            self.get_logger().info("카메라 데이터 수신 대기 중...")
            time.sleep(0.5)

        self.get_logger().info("카메라 데이터 수신 완료. 제어 루프를 가동합니다.")

        # 메인 주기 50Hz 제어 루프
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)

            # 1. 상태 전이 로직 수행 (FSM 의사결정 단일화)
            next_state = self._get_next_state()
            prev_state = self.current_drive_state
            
            if prev_state != next_state:
                self.current_drive_state = next_state
                self.get_logger().info(f"[STATE-TRANSITION] {prev_state} -> {next_state}")
                if next_state != DriveState.LANE_DRIVING:
                    self.is_curve_mode = False

            # 2. 상태별 조향 및 속도 명령 산출
            steer_cmd = 0.0
            speed_cmd = 0.0
            curvature = 0.0
            raw_curvature = 0.0
            is_sharp_curve = False

            if self.current_drive_state == DriveState.WAIT_FOR_GREEN:
                steer_cmd = 0.0
                speed_cmd = 0.0

            elif self.current_drive_state == DriveState.CONE_DRIVING:
                # 라바콘 회피 주행 (친구의 ConeDriver 사용)
                steer_cmd, speed_cmd = self.cone_driver.compute_steering(self.lidar_ranges)

                # 디버그 모니터 윈도우 표시 (카메라 이미지에 텍스트 합성)
                if self.image is not None:
                    debug_cone = self.image.copy()
                    cv2.putText(debug_cone, "State: CONE DRIVING (LiDAR)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    cv2.putText(debug_cone, f"Steer: {steer_cmd:.1f} | Speed: {speed_cmd:.1f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    cv2.imshow("Lane Detection Debug", debug_cone)
                    cv2.waitKey(1)

            elif self.current_drive_state == DriveState.LANE_DRIVING:
                # 속도 기반 동적 전방주시거리 계산
                lookahead_distance = max(self.lookahead_min, min(self.lookahead_max, 0.8 + 0.10 * self.motor_msg.speed))
                
                # BEV 상의 물리적 횡오차 e_y 및 전방 도로 곡률 계산 (곡선 모드 여부 전달)
                e_y, raw_curvature, debug_img = self.lane_detector.detect(self.image, lookahead_distance, self.is_curve_mode)
                
                # 곡률 지수 이동 평균(EMA) 필터 적용 (노이즈 억제)
                if raw_curvature is not None:
                    self.filtered_curvature = self.curv_alpha * raw_curvature + (1.0 - self.curv_alpha) * self.filtered_curvature
                else:
                    self.filtered_curvature = 0.0
                
                # 이후 판정에는 필터링된 곡률 사용
                curvature = self.filtered_curvature
                
                # Pure Pursuit 조향각 산출
                steer_cmd = self.pure_pursuit_steering(e_y, lookahead_distance)
                
                # 곡률 변화에 따른 이중 임계값 Hysteresis 상태 천이 및 급커브 판정
                if curvature is not None:
                    if curvature > self.curve_enter_threshold:
                        self.is_curve_mode = True
                        self.last_curve_time = time.time()
                    elif curvature > self.curve_exit_threshold:
                        self.last_curve_time = time.time()
                
                # 곡선 모드 해제 조건
                if self.is_curve_mode:
                    if curvature is not None and curvature < self.curve_exit_threshold:
                        if time.time() - self.last_curve_time > self.curve_exit_delay:
                            self.is_curve_mode = False
                
                is_sharp_curve = self.is_curve_mode
                
                if is_sharp_curve:
                    # 급커브 구간: 감속 (4.0m/s 고정) 및 강제 최대 조향
                    speed_cmd = self.speed_min
                else:
                    # 직선 및 완만한 커브: 조향 기반 동적 감속
                    curvature_steer_equiv = curvature * 0.8 if curvature is not None else 0.0
                    speed_steer_metric = max(abs(steer_cmd), curvature_steer_equiv)
                    speed_cmd = self._map_speed_by_steer(speed_steer_metric)
 
                # 차선 디버그 모니터 윈도우 갱신
                if debug_img is not None:
                    cv2.putText(debug_img, f"Lap: {self.lap_count}/{self.total_laps}", (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(debug_img, f"State: LANE | Curv: {raw_curvature:.1f} (Filt: {curvature:.1f}) ({'CURVE' if is_sharp_curve else 'STRAIGHT'})", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow("Lane Detection Debug", debug_img)
                    cv2.waitKey(1)

            elif self.current_drive_state == DriveState.PEDESTRIAN_STOP:
                steer_cmd = 0.0
                speed_cmd = self.speed_stop

            elif self.current_drive_state == DriveState.FINISHED:
                steer_cmd = 0.0
                speed_cmd = 0.0
                self.drive(angle=steer_cmd, speed=speed_cmd)
                break

            # 3. 조향 제한 및 필터 적용 (LANE_DRIVING 상태에서만 작동하며, CONE_DRIVING은 자체 필터 내장)
            if self.current_drive_state == DriveState.LANE_DRIVING:
                if is_sharp_curve:
                    # 급커브 시 PID 우회하여 모터가 즉시 최대 조향각으로 꺾임
                    if steer_cmd > 0:
                        steer_cmd = 100.0
                    elif steer_cmd < 0:
                        steer_cmd = -100.0
                    self.prev_steer = steer_cmd
                else:
                    # [Jejudol_ws 벤치마킹] Slew Rate Limiter + EMA 로우패스 필터
                    max_steer_change = 25.0
                    steer_diff = steer_cmd - self.prev_steer
                    steer_diff = max(-max_steer_change, min(max_steer_change, steer_diff))
                    rate_limited_steer = self.prev_steer + steer_diff
                    
                    steer_alpha = 0.60
                    steer_cmd = steer_alpha * rate_limited_steer + (1.0 - steer_alpha) * self.prev_steer
                    steer_cmd = max(-100.0, min(100.0, steer_cmd))
                    self.prev_steer = steer_cmd
            else:
                # 대기 또는 라바콘(CONE_DRIVING) 상태 등에서는 조향 임시 버퍼 유지
                self.prev_steer = steer_cmd

            # 4. 제어 명령 하위 구동계 전송
            self.drive(angle=steer_cmd, speed=speed_cmd)

            # 5. 실시간 주기 상태 모니터 로깅
            self._maybe_log_status(steer_cmd, speed_cmd, raw_curvature if raw_curvature is not None else 0.0)

            # 20ms 주기 (50Hz) 유지
            time.sleep(0.02)

def main(args=None):
    rclpy.init(args=args)
    node = TrackDriverNode()

    try:
        node.main_loop()
    except KeyboardInterrupt:
        node.get_logger().info("사용자 정지 요청(Ctrl+C)")
    finally:
        try:
            node.drive(angle=0, speed=0)
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(1)
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()
