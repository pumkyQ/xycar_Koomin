import math
import cv2
import numpy as np

class ConeDriver:
    def __init__(self, target_speed=20, kp=70.0, max_angle=100.0):
        """
        라바콘 회피 주행을 위한 클래스입니다.
        """
        self.target_speed = target_speed
        self.kp = kp
        self.max_angle = max_angle

    def compute_steering(self, lidar_ranges, cv_image=None):
        """
        라이다 배열과 카메라 이미지를 받아 조향각과 속도를 반환합니다.
        Sensor Fusion: Camera (60%) + LiDAR (40%)
        """
        if not lidar_ranges or len(lidar_ranges) < 50:
            return 0.0, 0.0

        # ==========================================================
        # 1. LiDAR Processing (클러스터링 및 차선 노이즈 필터링)
        # ==========================================================
        num_ranges = len(lidar_ranges)
        mid = num_ranges // 2
        hood_half = int(num_ranges * 0.08) 

        points = []
        angle_inc = math.radians(270.0 / num_ranges) # 505개 레이, 270도 기준
        
        for i in range(num_ranges):
            d = lidar_ranges[i]
            # 저 멀리 있는 고깔(최대 20m)까지 미리 인식하여 선을 그리도록 확장
            if not math.isfinite(d) or d <= 0.1 or d > 20.0: 
                continue
                
            # 시뮬레이터 360도 라이다 (angle_min=0, angle_inc=1도)
            # ROS 표준: 0도=전방, 90도=좌측, 180도=후방, 270도=우측
            # 우리 알고리즘 기준: 전방=Y(+), 좌측=X(+)
            angle_rad = math.radians(i)
            y = d * math.cos(angle_rad) # 전방
            x = d * math.sin(angle_rad) # 좌측
            
            # 후방 노이즈 무시 (Y < 0.3)
            if y < 0.3: continue
            
            # 차체 앞 코(후드) 간섭 무시 (전방 0도 주변)
            if (i < 20 or i > 340) and d <= 0.7:
                continue
            
            points.append((x, y))

        # --- 클러스터링 (가까운 점들을 묶어 하나의 고깔 객체로 생성) ---
        clusters = []
        if points:
            current_cluster = [points[0]]
            for i in range(1, len(points)):
                p1 = points[i-1]
                p2 = points[i]
                dist = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
                # 인접한 레이끼리의 거리가 40cm 이내면 같은 고깔 표면으로 간주
                if dist < 0.4:
                    current_cluster.append(p2)
                else:
                    cx = sum(p[0] for p in current_cluster) / len(current_cluster)
                    cy = sum(p[1] for p in current_cluster) / len(current_cluster)
                    clusters.append((cx, cy))
                    current_cluster = [p2]
            
            if current_cluster:
                cx = sum(p[0] for p in current_cluster) / len(current_cluster)
                cy = sum(p[1] for p in current_cluster) / len(current_cluster)
                clusters.append((cx, cy))

        # ==========================================================
        # 1. 2D 차선 연속 추적 알고리즘 (Nearest Neighbor)
        # ==========================================================
        def extract_lane(all_clusters, is_left):
            if not all_clusters: return []
            
            # 1. 시야 내의 가까운 고깔들 중 출발점 후보 색출 (선이 끝났을 때 저 멀리 있는 고깔을 새로 잡는 것 방지)
            front_cones = sorted([c for c in all_clusters if c[1] < 7.0 and abs(c[0]) < 8.0], key=lambda c: math.hypot(c[0], c[1]))
            if not front_cones: return []
            
            # 2. 가장 가까운 고깔(C1)과, 그 반대편 차선에 있을 법한 고깔(C2) 탐색
            C1 = front_cones[0]
            C2 = None
            for c in front_cones[1:]:
                # 가로(X)로 3m 이상 떨어져 있으면 반대편 차선으로 간주 (트랙 폭 5m)
                if abs(c[0] - C1[0]) > 3.0:
                    C2 = c
                    break
                    
            if C2 is not None:
                # 양쪽 차선이 모두 보일 때: X가 더 큰 쪽이 왼쪽 차선!
                left_start = C1 if C1[0] > C2[0] else C2
                right_start = C2 if C1[0] > C2[0] else C1
                start_c = left_start if is_left else right_start
            else:
                # 한쪽 차선만 보일 때: 완만한 코너에서 X축을 넘어갈 수 있으므로 기준 완화
                candidates = [c for c in front_cones if (c[0] > -1.5 if is_left else c[0] < 1.5)]
                if not candidates: return []
                start_c = min(candidates, key=lambda c: math.hypot(c[0], c[1]))
                
            lane = [start_c]
            remaining = list(all_clusters)
            remaining.remove(start_c)
            
            # 3. 2D 연속 추적 (징검다리)
            while remaining:
                last_c = lane[-1]
                
                neighbors = []
                for c in remaining:
                    # 절대 허용 불가: 왼쪽 차선이 오른쪽 깊숙이(X < -0.5) 침범하거나, 오른쪽이 왼쪽(X > 0.5) 침범하는 것 금지
                    if is_left and c[0] < -0.5:
                        continue
                    if not is_left and c[0] > 0.5:
                        continue
                        
                    dy = c[1] - last_c[1]
                    # Plot X 기준 변화량 (사용자 정의 dx): last_c[0] - c[0]
                    dx_plot = last_c[0] - c[0]
                    
                    # 고객님 요청사항: dx는 -2.5 ~ 0.5 사이, dy는 0.1 ~ 3.0 사이를 모두 만족할 때만 추가
                    if -2.5 <= dx_plot <= 0.5 and 0.1 <= dy <= 3.0:
                        neighbors.append(c)
                        
                if not neighbors:
                    break
                    
                # 코너에서는 우측 고깔도 X>0이 되므로 중앙(X=0) 기준 탐색은 위험함
                # 단순히 '이전 고깔에서 가장 물리적으로 가까운 고깔'을 다음 점으로 선택
                best_c = min(neighbors, key=lambda c: math.hypot(c[0] - last_c[0], c[1] - last_c[1]))
                
                lane.append(best_c)
                remaining.remove(best_c)
                
            return lane

        left_lane = extract_lane(clusters, is_left=True)
        right_lane = extract_lane(clusters, is_left=False)

        # ---------------------------------------------------------
        # 차선 점프(Crossover) 완벽 해결 로직
        # ---------------------------------------------------------
        # 1. 두 차선이 같은 고깔에서 시작했다면 (한쪽 차선만 보일 때), X 좌표 기준으로 진짜 주인을 판별
        if left_lane and right_lane and left_lane[0] == right_lane[0]:
            if left_lane[0][0] > 0:
                right_lane = []
            else:
                left_lane = []

        # 2. 주행 중 한 차선이 끊겨 반대편 차선 고깔로 점프한 경우, 더 자연스럽게 연결된(거리가 짧은) 쪽에 소유권을 줌
        if left_lane and right_lane:
            shared = set(left_lane) & set(right_lane)
            if shared:
                first_shared = min(shared, key=lambda c: c[1])
                l_idx = left_lane.index(first_shared)
                r_idx = right_lane.index(first_shared)
                
                l_dist = math.hypot(left_lane[l_idx][0] - left_lane[l_idx-1][0], left_lane[l_idx][1] - left_lane[l_idx-1][1]) if l_idx > 0 else 999
                r_dist = math.hypot(right_lane[r_idx][0] - right_lane[r_idx-1][0], right_lane[r_idx][1] - right_lane[r_idx-1][1]) if r_idx > 0 else 999
                
                if l_dist < r_dist:
                    right_lane = right_lane[:r_idx]
                else:
                    left_lane = left_lane[:l_idx]
        # ---------------------------------------------------------
        
        # ---------------------------------------------------------
        # 3. 고깔이 1개만 남았을 때, 반대편 차선의 곡률(기울기)을 활용하여 부드럽게 연장
        # ---------------------------------------------------------
        if len(left_lane) == 1 and len(right_lane) >= 2:
            dx = right_lane[1][0] - right_lane[0][0]
            dy = right_lane[1][1] - right_lane[0][1]
            left_lane.append((left_lane[0][0] + dx, left_lane[0][1] + dy))
            
        elif len(right_lane) == 1 and len(left_lane) >= 2:
            dx = left_lane[1][0] - left_lane[0][0]
            dy = left_lane[1][1] - left_lane[0][1]
            right_lane.append((right_lane[0][0] + dx, right_lane[0][1] + dy))

        # ==========================================================
        # 2. Piecewise Linear Interpolation (선형 보간 및 가상 중앙선 산출)
        # ==========================================================
        target_x = 0.0
        
        # 조기 조향 방지 및 코너링 최적화를 위해 전방주시거리를 기존 12m에서 7m 수준으로 현실화 (max(3.5, target_speed * 0.35))
        base_lookahead = max(3.5, self.target_speed * 0.35) 
        TRACK_HALF_WIDTH = 2.5 # 고깔이 좌우 2.5m (총 5m 폭)
        
        def fit_lane_x_at_y(lane_pts, target_y):
            if not lane_pts: return None
            
            # 고깔이 끝난 지점 1m 이후부터는 억지 예측(직선 연장)을 포기하고 None을 반환
            # 이렇게 하면 반대편 차선(살아있는 곡선)의 곡률을 그대로 복사해서 따라가게 됨!
            if target_y > lane_pts[-1][1] + 1.0:
                return None
                
            if len(lane_pts) == 1:
                return lane_pts[0][0]
                
            # 타겟 Y를 감싸는 앞뒤 두 점을 찾아 직선으로 연결
            for i in range(len(lane_pts) - 1):
                p1 = lane_pts[i]
                p2 = lane_pts[i+1]
                
                if p1[1] <= target_y <= p2[1]:
                    ratio = (target_y - p1[1]) / (p2[1] - p1[1]) if p2[1] != p1[1] else 0.0
                    return p1[0] + ratio * (p2[0] - p1[0])
                    
            # 타겟 Y가 첫 고깔보다 가까울 경우 (첫 두 점으로 직진 연장)
            if target_y < lane_pts[0][1]:
                p1, p2 = lane_pts[0], lane_pts[1]
                ratio = (target_y - p1[1]) / (p2[1] - p1[1]) if p2[1] != p1[1] else 0.0
                return p1[0] + ratio * (p2[0] - p1[0])
                
            # 타겟 Y가 마지막 고깔보다 멀 경우 (마지막 두 점으로 직진 연장)
            p1, p2 = lane_pts[-2], lane_pts[-1]
            ratio = (target_y - p1[1]) / (p2[1] - p1[1]) if p2[1] != p1[1] else 0.0
            return p1[0] + ratio * (p2[0] - p1[0])

        # 두 차선 중 가장 멀리 보이는 Y좌표를 확인하여 동적으로 타겟 Y 설정
        max_y = 0.0
        if left_lane: max_y = max(max_y, left_lane[-1][1])
        if right_lane: max_y = max(max_y, right_lane[-1][1])
        
        # 코너가 깊어서 고깔이 짤린 경우, 타겟점(Lookahead)을 그곳으로 당겨옴
        actual_lookahead = min(base_lookahead, max_y)
        if actual_lookahead < 1.0:
            actual_lookahead = 1.0

        pred_left_x = fit_lane_x_at_y(left_lane, actual_lookahead)
        pred_right_x = fit_lane_x_at_y(right_lane, actual_lookahead)
        
        if pred_left_x is not None and pred_right_x is not None:
            # 양쪽 차선이 모두 추정될 때 -> 두 선의 정중앙
            target_x = (pred_left_x + pred_right_x) / 2.0
        elif pred_left_x is not None:
            # 왼쪽 차선만 보일 때 -> 예측된 왼쪽 차선에서 2.5m 떨어진 곳
            target_x = pred_left_x - TRACK_HALF_WIDTH
        elif pred_right_x is not None:
            # 오른쪽 차선만 보일 때 -> 예측된 오른쪽 차선에서 2.5m 떨어진 곳
            target_x = pred_right_x + TRACK_HALF_WIDTH

        # ==========================================================
        # 3. 조향각 산출 (가상 중앙선 추종 - Pure Pursuit 기반 각도 제어)
        # ==========================================================
        if not hasattr(self, 'prev_angle'):
            self.prev_angle = 0.0

        # 타겟점(target_x)을 향한 측면 오차(Lateral Error) 계산
        error = target_x
        
        # 조향각 = 오차 * 비례상수(self.kp)
        # 타겟이 좌측(양수)일 때, 자이카는 좌회전 조향이 음수(-)이므로 부호 반전!
        raw_angle = - (error * self.kp) 
        
        # Low Pass Filter (민감도 완화, 스무딩)
        angle = self.prev_angle * 0.6 + raw_angle * 0.4
        self.prev_angle = angle
        
        # 중앙 추종을 위해서는 미세한 우측 조향이 필수적이므로 우회전 차단 해제!
        angle = max(-self.max_angle, min(self.max_angle, angle))

        # 감속 로직 (조향각이 커질 때 속도 비율 감속)
        speed = self.target_speed
        if abs(angle) > 50.0:
            speed = self.target_speed * 0.6
        elif abs(angle) > 20.0:
            speed = self.target_speed * 0.8

        return float(angle), float(speed)
