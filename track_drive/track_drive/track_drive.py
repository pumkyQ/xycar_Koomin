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
from track_drive.obstacle_detector import ObstacleDetector
from track_drive.road_sign_detector import RoadSignDetector, ZONE_SCHOOL
from track_drive.traffic_light_detector import (
    TrafficLightDetector, SIGNAL_GREEN, SIGNAL_RED,
    SIGNAL_YELLOW, SIGNAL_LEFT_ARROW, SIGNAL_UNKNOWN
)

#====================================================================
# [Drive States]
# 제주도 대회 FSM 설계 패턴을 벤치마킹한 3단계 핵심 주행 상태 정의
#====================================================================
class DriveState:
    WAIT_FOR_GREEN = 'wait_for_green'     # 1. 신호 대기 상태
    LANE_DRIVING = 'lane_driving'         # 2. Pure Pursuit 차선 주행 상태
    CONE_DRIVING = 'cone_driving'         # 3. 라바콘(문코스) 주행 상태
    PEDESTRIAN_STOP = 'pedestrian_stop'   # 4. 보행자 감지 긴급 정지 상태
    FINISHED = 'finished'                 # 3바퀴 완주 후 정지 상태

class TrackDriverNode(Node):
    def __init__(self):
        super().__init__('driver')
        self.get_logger().info('===== 국민대 대회 예선과제 1번 FSM 기반 제어 노드 시작 =====')

        # 센서 데이터 및 브릿지 초기화
        self.image = None
        self.lidar_ranges = None
        self.bridge = CvBridge()
        self.motor_msg = XycarMotor()

        # 인지 모듈 객체 초기화
        self.lane_detector = LaneDetector()
        self.traffic_detector = TrafficLightDetector()
        self.obstacle_detector = ObstacleDetector()

        # 주행 상태 머신 및 타이밍 초기화
        self.current_drive_state = DriveState.WAIT_FOR_GREEN
        self.lap_count = 0
        self.total_laps = 3

        # 퓨어퍼슛 파라미터 설정
        self.wheelbase = 0.33
        self.focal_length = 350.0
        self.steer_gain = 4.0       # 최근 피드백 조향 감도(4.0) 적용
        self.lookahead_min = 0.5    # 최소 전방주시거리 (급커브 대응)
        self.lookahead_max = 1.8    # 최대 전방주시거리 (직진 안정성 확보, 기존 1.3에서 상향)
        
        # 속도 기본값 설정 (동적 속도 제어 범위)
        self.speed_max = 10.0      # 직진 최고 속도
        self.speed_min = 3.5       # 커브 최저 속도 (최근 피드백 반영)
        self.speed_stop = 0.0       # 정지 속도
        
        # 보행자 안전 대기 관련 변수
        self.pedestrian_clear_time = 0.0
        self.recovery_delay_sec = 3.0  # 보행자 이탈 후 재출발까지 안전 대기 시간 (3초)

        # 바퀴 수 완주 측정을 위한 타이밍 변수
        self.start_time = time.time()
        self.last_lap_time = time.time()

        # 조향 PID 필터 파라미터 및 상태 변수
        self.steer_kp = 0.60
        self.steer_kd = 0.15
        self.steer_ki = 0.02
        self.prev_steer = 0.0
        self.prev_steer_error = 0.0
        self.steer_integral = 0.0

        # 콘 개수 실시간 디버그용 상태 변수
        self.detected_cones_count = 0
        self.last_cone_log_time = 0.0

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
        - angle: 사용자 입력 조향각 (물리 제어 범위: 좌측 최대 -100, 우측 최대 100)
        - speed: 주행 속도
        """
        # 조향이 최대로 꺾이도록 0.5배 스케일링을 제거하고 물리적 최대 조향각 적용
        clamped_angle = max(-100.0, min(100.0, angle))
        
        self.motor_msg.angle = float(clamped_angle)
        self.motor_msg.speed = float(speed)
        self.motor_pub.publish(self.motor_msg)

    #====================================================================
    # [Steering Controller]
    # Pure Pursuit 기하학적 제어기
    #====================================================================
    def pure_pursuit_steering(self, steer_offset):
        """
        Pure Pursuit 알고리즘을 사용하여 조향각을 계산합니다.
        - steer_offset: 차선 중심 오프셋 (픽셀 단위)
        """
        e_x = steer_offset / 0.4
        speed = self.motor_msg.speed
        
        # 속도 기반 동적 전방주시거리(Lookahead Distance) 계산 (직진 및 코너링 동적 튜닝)
        lookahead_distance = max(self.lookahead_min, min(self.lookahead_max, 0.4 + 0.08 * speed)) 
        
        # 퓨어퍼슛 조향각 계산 공식 (원근 투영 비례 축소 반영)
        delta = math.atan2(2.0 * self.wheelbase * e_x, self.focal_length * lookahead_distance)
        angle_deg = math.degrees(delta)
        
        # 조향 게인 반영 및 출력 범위 제한 (-100 ~ 100)
        steer_cmd = angle_deg * self.steer_gain
        return max(-100.0, min(100.0, steer_cmd))

    def _map_speed_by_steer(self, steer_cmd):
        """
        조향각의 절대값에 따라 차량 속도를 동적으로 매핑합니다.
        - 조향각이 0에 가까울수록 (직진): speed_max (10.0)
        - 조향각이 100에 가까울수록 (코너): speed_min (3.5)
        """
        steer_ratio = min(1.0, abs(steer_cmd) / 100.0)
        # 선형 감속 보간
        return self.speed_max - steer_ratio * (self.speed_max - self.speed_min)

    def _count_detected_cones(self):
        """
        LIDAR 데이터를 클러스터링하여 전방 4.0m 내 좌우 90도(총 180도) 이내 감지된 라바콘의 개수를 반환합니다.
        """
        if self.lidar_ranges is None:
            return 0

        ranges = np.array(self.lidar_ranges, dtype=np.float32)
        n = len(ranges)
        if n == 0:
            return 0

        angles_per_point = 360.0 / n
        points = []

        for i in range(n):
            angle_deg = i * angles_per_point
            if angle_deg > 180:
                angle_deg -= 360

            # 전방 좌우 90도 범위 및 4.0m 이내
            if -90 <= angle_deg <= 90:
                dist = ranges[i]
                if 0.1 < dist < 4.0 and np.isfinite(dist):
                    # 극좌표 -> 직교좌표 (x, y) 변환
                    angle_rad = math.radians(angle_deg)
                    x = dist * math.sin(angle_rad) # 가로
                    y = dist * math.cos(angle_rad) # 세로
                    points.append((x, y))

        if not points:
            return 0

        # 거리 기반 클러스터링 (DBSCAN 스타일)
        clusters = []
        for p in points:
            placed = False
            for c in clusters:
                rep = c[0]
                dist = math.sqrt((p[0] - rep[0])**2 + (p[1] - rep[1])**2)
                if dist < 0.25: # 25cm 이내면 동일한 라바콘으로 묶음
                    c.append(p)
                    placed = True
                    break
            if not placed:
                clusters.append([p])

        # 각 클러스터의 포인트 개수가 최소 2개 이상인 것만 유효한 라바콘으로 판정
        valid_cones = [c for c in clusters if len(c) >= 2]
        return len(valid_cones)

    def _is_road_in_front_0_5m(self):
        """
        차량 전방 약 0.5m 부근 (이미지 하단 중앙 영역)에 검은색 도로가 보이고,
        차선 인식 모듈에서 유효한 차선이 감지되는지 여부를 판단합니다.
        """
        if self.image is None:
            return False
        
        h, w = self.image.shape[:2]
        # 전방 0.5m 부근 ROI (하단 80% ~ 95% 행, 가로 중앙 35% ~ 65% 열)
        bottom_roi = self.image[int(h * 0.8):int(h * 0.95), int(w * 0.35):int(w * 0.65)]
        hsv = cv2.cvtColor(bottom_roi, cv2.COLOR_BGR2HSV)
        
        # 검은색/어두운 회색 도로 HSV 범위
        lower_road = np.array([0, 0, 30])
        upper_road = np.array([180, 50, 160])
        road_mask = cv2.inRange(hsv, lower_road, upper_road)
        
        total_pixels = bottom_roi.shape[0] * bottom_roi.shape[1]
        road_pixels = cv2.countNonZero(road_mask)
        road_ratio = road_pixels / total_pixels
        
        # 도로 면적이 30% 이상이고, 차선 검출기가 유효 차선을 보고 있을 때 True
        lane_visible = (self.lane_detector.no_lane_count == 0)
        return (road_ratio > 0.30) and lane_visible

    #====================================================================
    # [FSM State Transitions]
    # 제주도 대회 의사결정 노드의 계층적 상태 판단 패턴 적용
    #====================================================================
    def _get_next_state(self):
        """
        현재 차량 상태 및 센서 데이터를 바탕으로 계층적 우선순위에 맞추어 다음 상태를 결정합니다.
        """
        # 1. 완주 종료 상태 고정
        if self.current_drive_state == DriveState.FINISHED:
            return DriveState.FINISHED

        # 2. 신호등 대기 상태 처리
        if self.current_drive_state == DriveState.WAIT_FOR_GREEN:
            signal, _ = self.traffic_detector.detect(self.image)
            if signal == SIGNAL_GREEN:
                self.get_logger().info("★ 녹색 신호 감지! 주행을 시작합니다.")
                self.start_time = time.time()
                self.last_lap_time = time.time()
                
                # 시작 시 전방 라바콘 유무에 따라 라바콘/차선 주행 즉각 선택
                num_cones = self._count_detected_cones()
                self.detected_cones_count = num_cones
                if num_cones >= 4:
                    self.get_logger().info(f"★ 초기 주행 선택: 라바콘 {num_cones}개 감지 ➔ 문코스(CONE_DRIVING) 주행 시작")
                    return DriveState.CONE_DRIVING
                return DriveState.LANE_DRIVING
            return DriveState.WAIT_FOR_GREEN

        # 3. 완주 체크 (3바퀴 완주 시 FINISHED로 전이)
        self.check_lap_completion()
        if self.lap_count >= self.total_laps:
            self.get_logger().info("★★★ 3바퀴 완주 성공! 주행을 안전하게 종료합니다. ★★★")
            return DriveState.FINISHED

        # 4. 보행자 감지 판단
        ped_info = self.obstacle_detector.detect_pedestrian(self.image, self.lidar_ranges)

        # 주행 상태 중 보행자가 앞(CENTER)을 가로막으면 긴급정지 상태로 전환 (차선/라바콘 주행 공통 적용)
        if self.current_drive_state in (DriveState.LANE_DRIVING, DriveState.CONE_DRIVING):
            if ped_info['pedestrian_detected'] and ped_info['should_stop'] and ped_info['pedestrian_direction'] == 'CENTER':
                self.get_logger().warn("⚠ 전방 보행자 감지! 긴급 정지 상태로 전환합니다. -> PEDESTRIAN_STOP")
                self.pedestrian_clear_time = time.time()
                return DriveState.PEDESTRIAN_STOP

        # 보행자 긴급 정지 상태 제어
        if self.current_drive_state == DriveState.PEDESTRIAN_STOP:
            if ped_info['pedestrian_detected'] and ped_info['should_stop'] and ped_info['pedestrian_direction'] == 'CENTER':
                # 보행자가 아직 존재하면 안전 쿨다운 리셋
                self.pedestrian_clear_time = time.time()
                return DriveState.PEDESTRIAN_STOP
            else:
                # 보행자가 사라진 상태에서 회복 대기시간(3초) 경과 후 주행 복귀
                elapsed = time.time() - self.pedestrian_clear_time
                if elapsed < self.recovery_delay_sec:
                    return DriveState.PEDESTRIAN_STOP
                self.get_logger().info(f"★ 보행자 이탈 및 안전 대기 시간({self.recovery_delay_sec}초) 경과 완료 ➔ 주행 복귀")

        # 5. 라바콘(문코스)과 차선 주행의 분기 판단
        num_cones = self._count_detected_cones()
        self.detected_cones_count = num_cones

        # 1초에 한 번씩 콘 개수 실시간 디버그 로그 출력
        current_time = time.time()
        if current_time - self.last_cone_log_time >= 1.0:
            self.get_logger().info(f"[LIDAR DETECT] Cones counted in front: {num_cones}")
            self.last_cone_log_time = current_time
        
        # 현재 FSM 상태에 기반한 세부 분기 전이 제어
        if self.current_drive_state == DriveState.CONE_DRIVING:
            # 전방 0.5m에 검은색 도로가 나타나고, 디텍된 콘이 2개 이하가 되면 라인 주행으로 이탈 방지 복귀
            if num_cones <= 2 and self._is_road_in_front_0_5m():
                self.get_logger().info("★ 검은색 도로 감지 및 콘 2개 이하 ➔ 차선(LANE_DRIVING) 주행으로 안정 복귀")
                return DriveState.LANE_DRIVING
            return DriveState.CONE_DRIVING
        else:
            # 차선 주행 상태에서는 콘이 4개 이상으로 증대되면 콘 회피 주행 진입
            if num_cones >= 4:
                if self.current_drive_state != DriveState.CONE_DRIVING:
                    self.get_logger().info(f"★ 라바콘 {num_cones}개 감지 ➔ 문코스(CONE_DRIVING) 주행 시작")
                return DriveState.CONE_DRIVING
            return DriveState.LANE_DRIVING

    #====================================================================
    # [Lap Counter]
    # 신호등 재검출 기반 바퀴 수 카운트 및 완주 판단 알고리즘
    #====================================================================
    def check_lap_completion(self):
        """차량이 트랙 한 바퀴를 완주하여 출발 지점의 신호등을 다시 감지할 때를 판별합니다."""
        if self.current_drive_state in (DriveState.WAIT_FOR_GREEN, DriveState.FINISHED):
            return

        signal, _ = self.traffic_detector.detect(self.image)
        
        # 신호등 유효 감지 및 25초간의 쿨다운을 적용해 중복 카운트 방지
        if signal != SIGNAL_UNKNOWN:
            current_time = time.time()
            if current_time - self.last_lap_time > 25.0:
                self.lap_count += 1
                self.last_lap_time = current_time
                self.get_logger().info(f"★★★ Lap {self.lap_count} 완료! (소요 시간: {current_time - self.start_time:.1f}초) ★★★")

    def _maybe_log_status(self, steer_cmd, speed_cmd):
        """
        차량의 현재 주행 상태와 제어 토픽 값을 디버그 로깅합니다.
        """
        self.get_logger().info(
            f"[FSM STATUS] State: {self.current_drive_state} | "
            f"Steer: {steer_cmd:.1f} | Speed: {speed_cmd:.1f} | "
            f"Laps: {self.lap_count}/{self.total_laps} | "
            f"Cones: {self.detected_cones_count}"
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

            # 2. 상태별 조향 및 속도 명령 산출 (제어부와 상태 전이부의 독립성 보장)
            steer_cmd = 0.0
            speed_cmd = 0.0

            if self.current_drive_state == DriveState.WAIT_FOR_GREEN:
                steer_cmd = 0.0
                speed_cmd = 0.0

            elif self.current_drive_state == DriveState.LANE_DRIVING:
                steer_offset, debug_img = self.lane_detector.detect(self.image)
                steer_cmd = self.pure_pursuit_steering(steer_offset)
                # 조향각에 따른 동적 감속 매핑
                speed_cmd = self._map_speed_by_steer(steer_cmd)

                # 차선 디버그 모니터 윈도우 갱신
                if debug_img is not None:
                    cv2.putText(debug_img, f"Lap: {self.lap_count}/{self.total_laps}", (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(debug_img, f"State: LANE | Control: PURE_PURSUIT", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow("Lane Detection Debug", debug_img)
                    cv2.waitKey(1)

            elif self.current_drive_state == DriveState.CONE_DRIVING:
                cone_info = self.obstacle_detector.detect_cones(self.lidar_ranges)
                steer_cmd = cone_info['steer_offset'] * 2.0
                speed_cmd = 8.0  # 라바콘 회피 안전 기본 속도

            elif self.current_drive_state == DriveState.PEDESTRIAN_STOP:
                steer_cmd = 0.0
                speed_cmd = self.speed_stop

            elif self.current_drive_state == DriveState.FINISHED:
                steer_cmd = 0.0
                speed_cmd = 0.0
                self.drive(angle=steer_cmd, speed=speed_cmd)
                break

            # 3. 조향 PID 필터 적용 (LANE_DRIVING 및 CONE_DRIVING 상태에서만 동작)
            if self.current_drive_state in (DriveState.LANE_DRIVING, DriveState.CONE_DRIVING):
                target_steer = steer_cmd
                error = target_steer - self.prev_steer
                self.steer_integral = max(-50.0, min(50.0, self.steer_integral + error))
                d_error = error - self.prev_steer_error
                
                # PID 제어 법칙
                steer_cmd = self.prev_steer + (self.steer_kp * error) + (self.steer_ki * self.steer_integral) + (self.steer_kd * d_error)
                
                # 물리 범위 클램핑 적용 (Left max -100, Right max 100)
                steer_cmd = max(-100.0, min(100.0, steer_cmd))
                
                self.prev_steer = steer_cmd
                self.prev_steer_error = error
            else:
                # 대기 또는 정지 상태 등에서는 필터 상태 초기화
                self.prev_steer = steer_cmd
                self.prev_steer_error = 0.0
                self.steer_integral = 0.0

            # 4. 제어 명령 하위 구동계 전송
            self.drive(angle=steer_cmd, speed=speed_cmd)

            # 5. 실시간 주기 상태 모니터 로깅
            self._maybe_log_status(steer_cmd, speed_cmd)

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
        node.drive(angle=0, speed=0)
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
