"""학생 자율주차 알고리즘 스켈레톤 모듈.

수정 사항:
1. 맵 Y축 반전 처리 (상하 좌표 보정)
2. 장애물 감지 강화 (검은색 기둥/벽 인식)
3. 전면 주차 모드 자동 감지 및 적용
4. 경로 계획 안정성 개선
"""

import math
import heapq
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# --- 보조 함수 ---

def normalize_angle(angle: float) -> float:
    """각도를 -pi ~ pi 사이로 정규화합니다."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

def pretty_print_map_summary(map_payload: Dict[str, Any]) -> None:
    extent = map_payload.get("extent") or [None, None, None, None]
    slots = map_payload.get("slots") or []
    occupied = map_payload.get("occupied_idx") or []
    free_slots = len(slots) - sum(1 for v in occupied if v)
    
    print(f"[algo] map extent : {extent}")
    print(f"[algo] total slots: {len(slots)} / free: {free_slots}")
    
    stationary = map_payload.get("grid", {}).get("stationary")
    if stationary:
        rows = len(stationary)
        cols = len(stationary[0]) if len(stationary) > 0 else 0
        print(f"[algo] grid size  : {rows} x {cols}")

# --- 알고리즘용 클래스 정의 ---

@dataclass
class Node:
    """A* 탐색을 위한 노드"""
    x_idx: int
    y_idx: int
    yaw_idx: int
    x: float
    y: float
    yaw: float
    direction: int  # 1: 전진, -1: 후진
    cost: float = 0.0
    parent_index: Any = -1
    steer: float = 0.0

    def __lt__(self, other):
        return self.cost < other.cost

# --- PlannerSkeleton 구현 ---

@dataclass
class PlannerSkeleton:
    """경로 계획/제어 로직을 담는 클래스입니다."""

    map_data: Optional[Dict[str, Any]] = None
    map_extent: Optional[Tuple[float, float, float, float]] = None
    cell_size: float = 0.5
    stationary_grid: Optional[np.ndarray] = None
    waypoints: List[Tuple[float, float, float, int]] = field(default_factory=list)

    # 알고리즘 파라미터
    XY_RES: float = 0.5
    YAW_RES: float = math.radians(15.0)
    MOTION_RES: float = 0.5
    
    # 비용 가중치
    H_COST: float = 3.0
    REVERSE_COST: float = 2.0
    GEAR_SWITCH_COST: float = 20.0
    STEER_COST: float = 1.0

    # 주차 모드
    parking_mode: str = "rear_in"

    def __post_init__(self) -> None:
        if self.waypoints is None:
            self.waypoints = []

    def set_map(self, map_payload: Dict[str, Any]) -> None:
        """맵 데이터를 파싱하고 전처리합니다."""
        self.map_data = map_payload
        self.map_extent = tuple(
            map(float, map_payload.get("extent", (0.0, 0.0, 0.0, 0.0)))
        )
        self.cell_size = float(map_payload.get("cellSize", 0.5))
        
        # [중요] expected_orientation 필드로 주차 모드 자동 설정
        expected_orientation = map_payload.get("expected_orientation", "rear_in")
        if expected_orientation == "front_in":
            self.parking_mode = "front_in"
            print("[algo] Parking mode: FRONT-IN")
        else:
            self.parking_mode = "rear_in"
            print("[algo] Parking mode: REAR-IN")
        
        raw_grid = map_payload.get("grid", {}).get("stationary")
        if raw_grid:
            # [수정] Y축 상하 반전 처리
            self.stationary_grid = np.array(raw_grid)[::-1, :]
            
            # 장애물 팽창 (차량 크기 고려한 안전 마진)
            self._inflate_obstacles(radius=1.0)
            
        pretty_print_map_summary(map_payload)
        self.waypoints.clear()

    def _inflate_obstacles(self, radius: float):
        """장애물을 부풀려 충돌 안전성을 확보합니다."""
        if self.stationary_grid is None: return
        
        rows, cols = self.stationary_grid.shape
        inflated_grid = self.stationary_grid.copy()
        radius_idx = int(radius / self.cell_size)
        
        obstacles = np.argwhere(self.stationary_grid > 0.5)
        for r, c in obstacles:
            r_min = max(0, r - radius_idx)
            r_max = min(rows, r + radius_idx + 1)
            c_min = max(0, c - radius_idx)
            c_max = min(cols, c + radius_idx + 1)
            inflated_grid[r_min:r_max, c_min:c_max] = 1.0
            
        self.stationary_grid = inflated_grid
        print(f"[algo] Inflated obstacles by {radius}m")

    def _check_collision(self, x: float, y: float) -> bool:
        """좌표 충돌 검사"""
        if self.stationary_grid is None: return False
        
        xmin, _, ymin, _ = self.map_extent
        ix = int((x - xmin) / self.cell_size)
        iy = int((y - ymin) / self.cell_size)
        
        rows, cols = self.stationary_grid.shape
        if ix < 0 or ix >= cols or iy < 0 or iy >= rows:
            return True
        
        # Y축 반전 처리가 되었으므로 그대로 인덱싱
        if self.stationary_grid[iy][ix] > 0.5:
            return True
            
        return False

    def compute_path(self, obs: Dict[str, Any]) -> None:
        """전면/후진 주차를 모두 지원하는 Hybrid A* 경로 계획"""
        state = obs.get("state")
        target_slot = obs.get("target_slot")
        limits = obs.get("limits")

        if not state or not target_slot or not limits:
            return

        sx, sy, syaw = state['x'], state['y'], state['yaw']
        
        # 주차 슬롯 중심 계산
        tx = (target_slot[0] + target_slot[1]) / 2.0
        ty = (target_slot[2] + target_slot[3]) / 2.0
        
        # 슬롯 방향 판단
        width = abs(target_slot[1] - target_slot[0])
        height = abs(target_slot[3] - target_slot[2])
        
        # 슬롯이 세로로 긴 경우
        if height > width:
            slot_yaw = math.pi / 2.0 if ty > sy else -math.pi / 2.0
        else:
            slot_yaw = 0.0
        
        # 목표 방향 설정
        if self.parking_mode == "front_in":
            # 전면 주차: 차 앞부분이 슬롯 안쪽을 향함
            final_yaw = slot_yaw
        else:
            # 후진 주차: 차 뒷부분이 슬롯 안쪽을 향함 (머리는 반대)
            final_yaw = normalize_angle(slot_yaw + math.pi)

        print(f"[algo] Planning ({self.parking_mode}): ({sx:.1f}, {sy:.1f}) -> ({tx:.1f}, {ty:.1f}), yaw={final_yaw:.2f}")

        # A* 탐색 시작
        xmin, _, ymin, _ = self.map_extent
        start_node = Node(
            int((sx - xmin)/self.XY_RES), int((sy - ymin)/self.XY_RES), int(syaw/self.YAW_RES),
            sx, sy, syaw, 1
        )
        
        open_set = {(start_node.x_idx, start_node.y_idx, start_node.yaw_idx): start_node}
        closed_set = {}
        pq = [(0.0, (start_node.x_idx, start_node.y_idx, start_node.yaw_idx))]
        
        final_node = None
        max_iter = 100000  # 복잡한 맵을 위해 증가
        iter_count = 0

        while pq:
            cost, current_key = heapq.heappop(pq)
            if current_key in closed_set: continue
            
            current_node = open_set[current_key]
            closed_set[current_key] = current_node
            
            iter_count += 1
            if iter_count > max_iter:
                print("[algo] Max iteration reached.")
                break

            # 목표 도달 조건
            dist = math.hypot(current_node.x - tx, current_node.y - ty)
            angle_diff = abs(normalize_angle(current_node.yaw - final_yaw))
            
            if dist < 0.5 and angle_diff < math.radians(20.0):
                final_node = current_node
                print(f"[algo] Path found! Cost: {cost:.2f}, Iterations: {iter_count}")
                break

            # 노드 확장
            max_steer = limits.get('maxSteer', 0.6)
            steer_inputs = [-max_steer, -max_steer*0.5, 0, max_steer*0.5, max_steer]
            directions = [1, -1]
            
            for direction in directions:
                for steer in steer_inputs:
                    nx = current_node.x + direction * self.MOTION_RES * math.cos(current_node.yaw)
                    ny = current_node.y + direction * self.MOTION_RES * math.sin(current_node.yaw)
                    nyaw = normalize_angle(current_node.yaw + direction * self.MOTION_RES / limits.get('L', 2.7) * math.tan(steer))
                    
                    # [중요] 충돌 체크
                    if self._check_collision(nx, ny):
                        continue
                    
                    # 비용 계산
                    step_cost = self.MOTION_RES
                    if direction != current_node.direction: 
                        step_cost += self.GEAR_SWITCH_COST
                    if direction == -1: 
                        step_cost += self.REVERSE_COST
                    step_cost += abs(steer) * self.STEER_COST
                    
                    new_cost = current_node.cost + step_cost
                    
                    # 휴리스틱
                    heuristic = math.hypot(nx - tx, ny - ty) * self.H_COST
                    heuristic += abs(normalize_angle(nyaw - final_yaw)) * 2.0
                    
                    new_node = Node(
                        int((nx - xmin)/self.XY_RES), 
                        int((ny - ymin)/self.XY_RES), 
                        int(nyaw/self.YAW_RES),
                        nx, ny, nyaw, direction, new_cost, current_key, steer
                    )
                    
                    node_key = (new_node.x_idx, new_node.y_idx, new_node.yaw_idx)
                    
                    if node_key not in closed_set:
                        if node_key not in open_set or open_set[node_key].cost > new_cost:
                            open_set[node_key] = new_node
                            heapq.heappush(pq, (new_cost + heuristic, node_key))

        # 경로 재구성
        self.waypoints = []
        if final_node:
            curr = final_node
            while curr.parent_index != -1:
                self.waypoints.append((curr.x, curr.y, curr.yaw, curr.direction))
                curr = closed_set[curr.parent_index]
            self.waypoints.reverse()
            print(f"[algo] Generated {len(self.waypoints)} waypoints")
        else:
            print("[algo] Path finding FAILED. Using fallback.")
            self.waypoints = [(tx, ty, final_yaw, 1)]

    def compute_control(self, obs: Dict[str, Any]) -> Dict[str, float]:
        """Pure Pursuit 제어기"""
        if not self.waypoints:
            return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}

        state = obs["state"]
        limits = obs["limits"]
        curr_x, curr_y, curr_yaw = state['x'], state['y'], state['yaw']
        curr_v = state['v']
        L = limits.get('L', 2.7)

        # Lookahead 거리 계산
        lookahead_dist = 1.0 + 0.3 * abs(curr_v)
        
        # 가장 가까운 경로점 찾기
        min_d = float('inf')
        closest_idx = 0
        for i, (wx, wy, _, _) in enumerate(self.waypoints):
            d = math.hypot(wx - curr_x, wy - curr_y)
            if d < min_d:
                min_d = d
                closest_idx = i
                
        # 목표 지점 찾기
        target_idx = closest_idx
        for i in range(closest_idx, len(self.waypoints)):
            wx, wy, _, _ = self.waypoints[i]
            if math.hypot(wx - curr_x, wy - curr_y) > lookahead_dist:
                target_idx = i
                break
                
        tx, ty, _, tdir = self.waypoints[target_idx]
        
        # 조향각 계산
        alpha = normalize_angle(math.atan2(ty - curr_y, tx - curr_x) - curr_yaw)
        if tdir == -1:
             alpha = normalize_angle(math.atan2(ty - curr_y, tx - curr_x) - curr_yaw + math.pi)
             
        steer = math.atan2(2.0 * L * math.sin(alpha), lookahead_dist)
        steer = max(min(steer, limits['maxSteer']), -limits['maxSteer'])
        
        # 속도 프로파일
        dist_remain = math.hypot(self.waypoints[-1][0] - curr_x, self.waypoints[-1][1] - curr_y)
        
        target_speed = 2.0
        if abs(steer) > 0.3: target_speed = 1.0
        if dist_remain < 4.0: target_speed = 1.0
        if dist_remain < 1.5: target_speed = 0.5
        if dist_remain < 0.2: target_speed = 0.0
        
        if tdir == -1: target_speed = -target_speed
        
        # PID 제어
        err_v = abs(target_speed) - abs(curr_v)
        accel, brake = 0.0, 0.0
        
        gear = "D" if tdir == 1 else "R"
        
        if (gear == "D" and curr_v < -0.05) or (gear == "R" and curr_v > 0.05):
            brake = 1.0
        else:
            if err_v > 0:
                accel = min(limits['maxAccel'] * 0.5, err_v * 0.4)
            else:
                brake = min(limits['maxBrake'], -err_v * 0.8)
                
            if dist_remain < 0.1:
                accel = 0.0
                brake = 1.0

        return {"steer": float(steer), "accel": float(accel), "brake": float(brake), "gear": gear}

# 전역 인스턴스
planner = PlannerSkeleton()

def handle_map_payload(map_payload: Dict[str, Any]) -> None:
    planner.set_map(map_payload)

def planner_step(obs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if not planner.waypoints:
            planner.compute_path(obs)
        return planner.compute_control(obs)
    except Exception as exc:
        print(f"[algo] error: {exc}")
        import traceback
        traceback.print_exc()
        return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}
