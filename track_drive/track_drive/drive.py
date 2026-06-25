#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
import time
import cv2
import numpy as np
from rclpy.node import Node
from xycar_msgs.msg import XycarMotor
from sensor_msgs.msg import Image, LaserScan
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge

# 자체 모듈 임포트
from track_drive.lane_detector import LaneDetector
from track_drive.obstacle_detector import ObstacleDetector

#====================================================================
# [Traffic Light Detector]
# 예선과제 1번: 녹색 신호등 인식을 위한 경량 클래스
#====================================================================
class GreenLightDetector:
    def __init__(self):
        # ROI 설정 비율 (이미지 상단부에서 신호등 검출)
        self.roi_top = 0
        self.roi_bottom_ratio = 0.35  # 이미지 높이의 35%까지 사용
        self.roi_left_ratio = 0.1     # 이미지 좌측 10% 제외
        self.roi_right_ratio = 0.9    # 이미지 우측 10% 제외

        # HSV 색상 영역 정의 (녹색 신호등 기준)
        # 환경에 맞게 임계값을 조정해야 할 수 있습니다.
        self.grn_h_low, self.grn_h_high = 40, 90
        self.grn_s_min, self.grn_v_min = 80, 150

        # 최소 영역 크기 필터링 (노이즈 방지)
        self.min_area_threshold = 15

        # 디바운싱(연속 감지 검증)을 위한 카운터 변수
        self.green_count = 0
        self.confirm_threshold = 3  # 3프레임 연속 감지 시 최종 결정

    def is_green_light(self, image):
        if image is None:
            return False

        h, w = image.shape[:2]
        
        # ROI 크롭
        roi_bottom = int(h * self.roi_bottom_ratio)
        roi_left = int(w * self.roi_left_ratio)
        roi_right = int(w * self.roi_right_ratio)
        roi = image[self.roi_top:roi_bottom, roi_left:roi_right]

        # HSV 변환 및 이진화(Mask) 생성
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_green = np.array([self.grn_h_low, self.grn_s_min, self.grn_v_min])
        upper_green = np.array([self.grn_h_high, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # 윤곽선 검출을 통해 원형(혹은 신호등 형상) 판단
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_detected = False

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area_threshold:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = bw / max(bh, 1)

            # 신호등 원형 비율 검증 (가로세로 비율이 비교적 정방형에 가까운 범위)
            if 0.4 <= aspect_ratio <= 2.5:
                green_detected = True
                break

        # 신뢰성 강화를 위한 카운터 업데이트 (디바운싱)
        if green_detected:
            self.green_count += 1
        else:
            self.green_count = max(0, self.green_count - 1)

        # 일정 횟수 연속 검출되면 최종 녹색 신호등으로 인정
        if self.green_count >= self.confirm_threshold:
            return True
        return False


#====================================================================
# [Drive Node]
# 상태 머신 구조의 자율주행 메인 제어 노드
#====================================================================
class DriveState:
    WAIT_FOR_GREEN = 0  # 녹색 신호등 대기 상태
    GO_STRAIGHT = 1     # 출발 직후 단거리 직진 상태 (신호등 포스트 회피용)
    CONE_DRIVING = 2    # 라바콘 구간 주행 상태 (LiDAR 기반)
    LANE_DRIVING = 3    # 차선 주행 상태 (카메라 기반)
    FINISHED = 4        # 미션 완료 상태

class TrackDriverNode(Node):
    def __init__(self):
        super().__init__('drive')
        self.get_logger().info('===== Xycar 자율주행 drive 노드 시작 =====')

        # 센서 데이터 및 브릿지 초기화
        self.image = None
        self.lidar_ranges = None
        self.bridge = CvBridge()
        self.motor_msg = XycarMotor()

        # 각 검출 및 주행 모델 초기화
        self.green_detector = GreenLightDetector()
        self.lane_detector = LaneDetector()
        self.obstacle_detector = ObstacleDetector()

        # PID 제어용 변수
        self.pid_error_sum = 0.0
        self.pid_prev_error = 0.0

        # 주행 제어 방식 선택 ("PID" 또는 "PURE_PURSUIT")
        self.control_method = "PURE_PURSUIT"

        # 주행 상태 머신 초기 변수 설정
        self.drive_state = DriveState.WAIT_FOR_GREEN
        self.go_straight_start_time = 0.0
        self.go_straight_duration = 1.2  # 초 단위 (신호등 포스트를 통과하기 위한 직진 시간)
        self.cone_no_detect_count = 0
        self.last_cone_angle = 0.0

        # ROS2 토픽 발행 및 구독 설정
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
        - angle: 조향각 (-50 ~ 50)
        - speed: 속도
        """
        angle = max(-50.0, min(50.0, angle))
        self.motor_msg.angle = float(angle)
        self.motor_msg.speed = float(speed)
        self.motor_pub.publish(self.motor_msg)

    def pid_steering(self, error):
        """
        PID 제어기를 거쳐 조향각을 계산합니다.
        """
        kp, ki, kd = 0.5, 0.001, 0.3
        self.pid_error_sum += error
        self.pid_error_sum = max(-500.0, min(500.0, self.pid_error_sum))
        d_error = error - self.pid_prev_error
        self.pid_prev_error = error

        angle = (kp * error) + (ki * self.pid_error_sum) + (kd * d_error)
        return angle

    def pure_pursuit_steering(self, steer_offset):
        """
        Pure Pursuit 알고리즘을 사용해 조향각을 계산합니다.
        - steer_offset: 차선 중심 오프셋 (픽셀 단위, lane_detector가 STEER_GAIN을 곱해 준 값)
        """
        # lane_detector는 내부에서 STEER_GAIN(0.4)을 곱해 리턴하므로 raw 픽셀 오차(e_x) 복원
        e_x = steer_offset / 0.4
        
        # 차량 기하학적 파라미터
        wheelbase = 0.33       # 축거 (L, 단위: m)
        focal_length = 350.0   # 카메라 가로 초점 거리 (pixels)
        
        # 현재 속도를 가져와 룩어헤드 거리(L_d, 단위: m)를 동적으로 계산
        # 속도가 빠를 때는 멀리보고(L_d 증가), 느릴 때는 가까이봅니다(L_d 감소)
        speed = self.motor_msg.speed
        lookahead_distance = max(0.5, min(1.8, 0.4 + 0.08 * speed)) 
        
        # Pure Pursuit 공식 적용 (라디안 단위)
        # delta = arctan(2 * L * e_x / (f_x * L_d))
        delta = np.arctan2(2.0 * wheelbase * e_x, focal_length * lookahead_distance)
        
        # 라디안을 도(degree) 단위로 변환
        angle_deg = np.degrees(delta)
        
        # 조향 보정 계수 (시뮬레이터 반응성에 맞게 미세 조정)
        steer_gain = 1.8
        angle = angle_deg * steer_gain
        
        return angle

    def handle_wait_for_green(self):
        """
        녹색 신호등이 켜질 때까지 대기하며 정지 상태를 유지합니다.
        """
        if self.green_detector.is_green_light(self.image):
            self.get_logger().info("★ 녹색 신호 감지! 출발합니다! (WAIT_FOR_GREEN -> GO_STRAIGHT)")
            self.drive_state = DriveState.GO_STRAIGHT
            self.go_straight_start_time = time.time()
        else:
            self.get_logger().info("신호 대기 중...")
            self.drive(angle=0, speed=0)

    def handle_go_straight(self):
        """
        출발 직후 신호등 포스트나 가이드 바 등의 구조물을 라바콘으로 
        오인식하여 오조향되는 것을 막기 위해 짧은 시간 동안 직진합니다.
        """
        elapsed = time.time() - self.go_straight_start_time
        if elapsed < self.go_straight_duration:
            self.drive(angle=0.0, speed=8.0)
        else:
            self.get_logger().info("★ 시작 구간 직진 해제 → 라바콘 감지 주행 전환 (GO_STRAIGHT -> CONE_DRIVING)")
            self.drive_state = DriveState.CONE_DRIVING

    def handle_cone_driving(self):
        """
        라이다를 바탕으로 라바콘 구간을 돌파합니다.
        라바콘이 더 이상 감지되지 않으면 차선 주행으로 자동 전환합니다.
        """
        cone_info = self.obstacle_detector.detect_cones(self.lidar_ranges)

        if cone_info['cone_detected']:
            angle = cone_info['steer_offset']
            self.drive(angle=angle, speed=8.0)
            self.cone_no_detect_count = 0
            self.last_cone_angle = angle
        else:
            # 라바콘이 검출되지 않는 프레임 누적 계산
            self.cone_no_detect_count += 1
            if self.cone_no_detect_count > 30:  # 약 0.6초 동안 라바콘이 없으면 통과한 것으로 간주
                self.get_logger().info("★ 라바콘 주행 완료 → 차선 주행 모드 전환 (CONE_DRIVING -> LANE_DRIVING)")
                self.drive_state = DriveState.LANE_DRIVING
                self.cone_no_detect_count = 0
            else:
                # 미감지 시 이전 조향각을 잠시 유지
                self.drive(angle=self.last_cone_angle, speed=8.0)

    def handle_lane_driving(self):
        """
        차선 인식 모듈을 통해 계산된 steer_offset을 조향에 반영하여 차선 주행을 수행합니다.
        """
        steer_offset, _, debug_img = self.lane_detector.detect(self.image)
        
        # 선택된 제어 알고리즘 적용
        if self.control_method == "PURE_PURSUIT":
            angle = self.pure_pursuit_steering(steer_offset)
        else:
            angle = self.pid_steering(steer_offset)
        
        # 차선 주행 속도 설정
        self.drive(angle=angle, speed=10.0)

        # 디버그 윈도우 표시 (원하는 경우 활성화)
        if debug_img is not None:
            cv2.imshow("Lane Detection Debug", debug_img)
            cv2.waitKey(1)

    def main_loop(self):
        # 카메라 데이터 수신 시까지 대기
        while rclpy.ok() and self.image is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            self.get_logger().info("카메라 영상 데이터 대기 중...")
            time.sleep(0.5)

        self.get_logger().info("카메라 데이터 수신 완료. 주행 제어 루프를 시작합니다.")

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)

            try:
                if self.drive_state == DriveState.WAIT_FOR_GREEN:
                    self.handle_wait_for_green()

                elif self.drive_state == DriveState.GO_STRAIGHT:
                    self.handle_go_straight()

                elif self.drive_state == DriveState.CONE_DRIVING:
                    self.handle_cone_driving()

                elif self.drive_state == DriveState.LANE_DRIVING:
                    self.handle_lane_driving()

                elif self.drive_state == DriveState.FINISHED:
                    self.drive(angle=0, speed=0)
                    break

            except Exception as e:
                self.get_logger().error(f"제어 중 에러 발생: {e}")
                self.drive(angle=0, speed=0)

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
