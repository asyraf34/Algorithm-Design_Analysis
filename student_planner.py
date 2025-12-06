"""학생 자율주차 알고리즘 스켈레톤 모듈.

이 파일만 수정하면 되고, 네트워킹/IPC 관련 코드는 `ipc_client.py`에서
자동으로 처리합니다. 학생은 아래 `PlannerSkeleton` 클래스나 `planner_step`
함수를 원하는 로직으로 교체/확장하면 됩니다.
"""

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np

# --- 보조 함수 ---
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
        cols = len(stationary[0]) if rows > 0 else 0
        print(f"[algo] grid size  : {rows} x {cols}")

def normalize_angle(angle):
    """각도를 -pi ~ pi 사이로 정규화"""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

# --- 알고리즘 노드 클래스 ---
@dataclass
class Node:
    """Hybrid A* 탐색을 위한 노드"""
    x_idx: int
    y_idx: int
    yaw_idx: int
    x: float
    y: float
    yaw: float
    direction: int # 1: Forward, -1: Reverse
    steer: float
    cost: float = 0.0
    parent_idx: Any = -1

    def __lt__(self, other):
        return self.cost < other.cost

# --- PlannerSkeleton (통합 구현) ---
@dataclass
class PlannerSkeleton:
    """경로 계획 및 제어 로직을 모두 포함하는 단일 클래스입니다."""

    # 1. 기본 맵 데이터
    map_data: Optional[Dict[str, Any]] = None
    map_extent: Optional[Tuple[float, float, float, float]] = None
    cell_size: float = 0.5
    stationary_grid: Optional[np.ndarray] = None
    waypoints: List[Tuple[float, float, float, int]] = field(default_factory=list)

    # 2. Hybrid A* 파라미터 (튜닝됨)
    XY_RES: float = 0.5       
    YAW_RES: float = math.radians(15.0)
    MOTION_RES: float = 0.6   
    N_STEER: int = 12         
    
    # 3. 비용 가중치
    H_COST: float = 2.5       
    R_COST: float = 1.5       
    S_COST: float = 2.0       
    GS_COST: float = 20.0     

    # 4. 제어 상태 변수
    expected_orientation: str = "front_in"
    current_target_idx: int = 0

    def __post_init__(self) -> None:
        if self.waypoints is None:
            self.waypoints = []

    def set_map(self, map_payload: Dict[str, Any]) -> None:
        """시뮬레이터에서 전송한 정적 맵 데이터를 보관합니다."""
        self.map_data = map_payload
        self.map_extent = tuple(
            map(float, map_payload.get("extent", (0.0, 0.0, 0.0, 0.0)))
        )
        self.cell_size = float(map_payload.get("cellSize", 0.5))
        
        raw_grid = map_payload.get("grid", {}).get("stationary") or []
        
        # [중요] 맵 데이터 상하 반전
        self.stationary_grid = np.array([list(map(float, row)) for row in raw_grid][::-1])
        
        self.expected_orientation = map_payload.get("expected_orientation", "front_in")
        
        pretty_print_map_summary(map_payload)
        self.waypoints.clear()
        print(f"[algo] Map set. Expected orientation: {self.expected_orientation}")

        # 2. 장애물 추가 처리
        self._rasterize_static_objects(map_payload)

        # 3. 장애물 팽창 (1.0m)
        self.inflate_obstacles(radius_m=1.0) 

    def _rasterize_static_objects(self, map_payload):
        """벽과 주차선을 그리드에 1.0으로 표시"""
        if self.stationary_grid is None: return
        
        def mark_rect(rect):
            min_x, _, min_y, _ = self.map_extent
            x0, x1 = sorted((float(rect[0]), float(rect[1])))
            y0, y1 = sorted((float(rect[2]), float(rect[3])))
            
            start_x = max(0, int((x0 - min_x) / self.cell_size))
            end_x = min(self.stationary_grid.shape[1] - 1, int(math.ceil((x1 - min_x) / self.cell_size)))
            
            raw_start_y = int((y0 - min_y) / self.cell_size)
            raw_end_y = int(math.ceil((y1 - min_y) / self.cell_size))
            
            rows = self.stationary_grid.shape[0]
            start_y = max(0, rows - 1 - raw_end_y)
            end_y = min(rows - 1, rows - 1 - raw_start_y)

            self.stationary_grid[start_y:end_y+1, start_x:end_x+1] = 1.0

        slots = map_payload.get("slots") or []
        occupied = map_payload.get("occupied_idx") or []
        for idx, is_occ in enumerate(occupied):
            if idx < len(slots) and is_occ:
                mark_rect(slots[idx])

        for wall in map_payload.get("walls_rects", []) or []:
            mark_rect(wall)
            
        line_thick = 0.2
        for line in map_payload.get("lines", []) or []:
            if len(line) == 4:
                x0, y0, x1, y1 = map(float, line)
                rect = (min(x0,x1)-line_thick, max(x0,x1)+line_thick, 
                        min(y0,y1)-line_thick, max(y0,y1)+line_thick)
                mark_rect(rect)

    def inflate_obstacles(self, radius_m: float = 1.0) -> None:
        """장애물 팽창"""
        if self.stationary_grid is None: return
        
        rows, cols = self.stationary_grid.shape
        radius_cells = max(1, int(math.ceil(radius_m / self.cell_size)))
        
        inflated = self.stationary_grid.copy()
        obstacles = np.argwhere(self.stationary_grid > 0.5)
        
        for r, c in obstacles:
            r_min = max(0, r - radius_cells)
            r_max = min(rows, r + radius_cells + 1)
            c_min = max(0, c - radius_cells)
            c_max = min(cols, c + radius_cells + 1)
            inflated[r_min:r_max, c_min:c_max] = 1.0
            
        self.stationary_grid = inflated
        print(f"[algo] Inflated obstacles by {radius_m}m ({radius_cells} cells)")

    # ---------------- 충돌 검사 ----------------
    def _check_collision(self, x, y, xmin, ymin):
        """좌표 (x, y) 충돌 확인"""
        if self.stationary_grid is None: return True
        
        gx = int((x - xmin) / self.cell_size)
        raw_gy = int((y - ymin) / self.cell_size)
        rows, cols = self.stationary_grid.shape
        gy = rows - 1 - raw_gy 
        
        if 0 <= gy < rows and 0 <= gx < cols:
            if self.stationary_grid[gy][gx] > 0.5:
                return False # Collision
        return True # Safe

    # ---------------- Helper Methods ----------------
    def _calc_idx(self, val, offset, is_yaw=False):
        res = self.YAW_RES if is_yaw else self.XY_RES
        return int(round((val - offset) / res))
    
    def _calc_index_id_val(self, x, y, yaw, x_offset, y_offset):
        xi = self._calc_idx(x, x_offset)
        yi = self._calc_idx(y, y_offset)
        yawi = self._calc_idx(yaw, -math.pi, True)
        return (xi, yi, yawi)

    def _calc_heuristic(self, nx, ny, tx, ty):
        return math.hypot(nx - tx, ny - ty)

    # ---------------- Path Planning ----------------
    def compute_path(self, obs: Dict[str, Any]) -> None:
        state = obs.get("state")
        target_slot = obs.get("target_slot")
        limits = obs.get("limits")
        
        if not state or not target_slot or not limits: return

        sx, sy, syaw = state['x'], state['y'], state['yaw']
        tx = (target_slot[0] + target_slot[1]) / 2.0
        ty = (target_slot[2] + target_slot[3]) / 2.0
        
        # 목표 방향 설정
        slot_w = target_slot[1] - target_slot[0]
        slot_h = target_slot[3] - target_slot[2]
        if slot_h > slot_w: 
             tyaw = math.pi / 2.0 if ty > 0 else -math.pi / 2.0
        else: 
             tyaw = 0.0

        print(f"[algo] Planning ({sx:.1f}, {sy:.1f}) -> ({tx:.1f}, {ty:.1f})...")
        xmin, xmax, ymin, ymax = self.map_extent

        # 진입점 설정
        approach_dist = 4.0 
        ax = tx - approach_dist * math.cos(tyaw)
        ay = ty - approach_dist * math.sin(tyaw)
        
        # A* Init
        open_set = {} 
        closed_set = {} 
        pq = [] 

        start_node = Node(
            self._calc_idx(sx, xmin), self._calc_idx(sy, ymin), self._calc_idx(syaw, -math.pi, True),
            sx, sy, syaw, 1, 0.0, 0.0, -1
        )
        
        start_id = (start_node.x_idx, start_node.y_idx, start_node.yaw_idx)
        open_set[start_id] = start_node
        heapq.heappush(pq, (0.0, start_id))

        max_iter = 60000 
        iter_count = 0
        final_node = None

        while pq:
            cost, curr_id = heapq.heappop(pq)
            if curr_id in closed_set: continue
            
            curr_node = open_set[curr_id]
            closed_set[curr_id] = curr_node
            
            if iter_count > max_iter:
                print("[algo] Max iter reached.")
                break
            iter_count += 1

            # Goal Check
            dist_to_goal = math.hypot(curr_node.x - tx, curr_node.y - ty)
            yaw_diff = abs(normalize_angle(curr_node.yaw - tyaw))
            
            if dist_to_goal < 0.5 and yaw_diff < math.radians(20): 
                print(f"[algo] Path found! Cost: {curr_node.cost:.2f}, Iter: {iter_count}")
                final_node = curr_node
                break

            # Neighbors Expansion
            max_steer = limits['maxSteer']
            steer_inputs = [-max_steer, -max_steer*0.6, 0, max_steer*0.6, max_steer]
            directions = [1, -1]
            
            for direction in directions:
                for steer in steer_inputs:
                    length = self.MOTION_RES
                    nx = curr_node.x + direction * length * math.cos(curr_node.yaw)
                    ny = curr_node.y + direction * length * math.sin(curr_node.yaw)
                    nyaw = normalize_angle(curr_node.yaw + direction * length / limits['L'] * math.tan(steer))

                    if not (xmin <= nx <= xmax and ymin <= ny <= ymax): continue
                    if not self._check_collision(nx, ny, xmin, ymin): continue 

                    # Cost Logic
                    step_cost = self.H_COST * length
                    if direction == -1: step_cost += length * self.R_COST
                    if direction != curr_node.direction: step_cost += self.GS_COST
                    step_cost += abs(steer) * self.S_COST

                    new_cost = curr_node.cost + step_cost
                    h_val = self._calc_heuristic(nx, ny, tx, ty)
                    if h_val < 5.0: h_val += abs(normalize_angle(nyaw - tyaw)) * 2.0

                    priority = new_cost + h_val
                    node_idx_id = self._calc_index_id_val(nx, ny, nyaw, xmin, ymin)

                    if node_idx_id in closed_set: continue
                    if node_idx_id not in open_set or open_set[node_idx_id].cost > new_cost:
                        new_node = Node(
                            self._calc_idx(nx, xmin), self._calc_idx(ny, ymin), self._calc_idx(nyaw, -math.pi, True),
                            nx, ny, nyaw, direction, steer, new_cost, curr_id
                        )
                        open_set[node_idx_id] = new_node
                        heapq.heappush(pq, (priority, node_idx_id))

        # Reconstruct
        self.waypoints = []
        if final_node:
            curr = final_node
            while curr.parent_idx != -1:
                self.waypoints.append((curr.x, curr.y, curr.yaw, curr.direction))
                curr = closed_set[curr.parent_idx]
            self.waypoints.reverse()
            self.current_target_idx = 0
            print(f"[algo] Path generated ({len(self.waypoints)} pts).")
        else:
            # 실패 시 비상 직진 (수정됨: goal 대신 tx, ty 사용)
            print("[algo] Hybrid A* failed. Generating fallback path.")
            self.waypoints = [(tx, ty, tyaw, 1)]

    # ---------------- Control Logic ----------------
    def compute_control(self, obs: Dict[str, Any]) -> Dict[str, float]:
        if not self.waypoints:
            return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}

        state = obs["state"]
        limits = obs["limits"]
        curr_x, curr_y, curr_yaw = state['x'], state['y'], state['yaw']
        curr_v = state['v']

        look_ahead_base = 1.5
        look_ahead_dist = max(look_ahead_base, min(4.0, abs(curr_v) * 1.0))
        
        closest_dist = float('inf')
        closest_idx = self.current_target_idx
        
        search_limit = min(len(self.waypoints), self.current_target_idx + 50)
        for i in range(self.current_target_idx, search_limit):
            d = math.hypot(self.waypoints[i][0] - curr_x, self.waypoints[i][1] - curr_y)
            if d < closest_dist:
                closest_dist = d
                closest_idx = i
        self.current_target_idx = closest_idx

        target_idx = closest_idx
        for i in range(closest_idx, len(self.waypoints)):
            d = math.hypot(self.waypoints[i][0] - curr_x, self.waypoints[i][1] - curr_y)
            if d > look_ahead_dist:
                target_idx = i
                break
        
        tx, ty, _, tdir = self.waypoints[target_idx]
        gear = "D" if tdir == 1 else "R"
        
        dx = tx - curr_x
        dy = ty - curr_y
        
        local_x = math.cos(curr_yaw) * dx + math.sin(curr_yaw) * dy
        local_y = -math.sin(curr_yaw) * dx + math.cos(curr_yaw) * dy
        
        if gear == "R": 
            local_x = -local_x
            local_y = -local_y
            
        ld2 = local_x**2 + local_y**2
        steer = 0.0
        if ld2 > 0.01:
            kappa = 2 * local_y / ld2
            steer = math.atan(kappa * limits['L'])
        
        # 속도 제어
        dist_final = math.hypot(self.waypoints[-1][0] - curr_x, self.waypoints[-1][1] - curr_y)
        
        if dist_final < 1.0: target_v = 0.5
        elif dist_final < 3.0: target_v = 1.5
        else: target_v = 3.0
        
        if gear == "R": target_v = -target_v
        
        err_v = abs(target_v) - abs(curr_v)
        accel, brake = 0.0, 0.0
        
        if target_speed := target_v: # 변수명 통일
             pass

        if abs(target_v) < 0.1 and abs(curr_v) < 0.1: brake = 1.0
        elif err_v > 0: accel = min(limits['maxAccel'], 0.8 * err_v)
        else: brake = min(limits['maxBrake'], 0.2)

        if (gear == "D" and curr_v < -0.05) or (gear == "R" and curr_v > 0.05):
            accel = 0.0
            brake = 1.0

        return {"steer": float(steer), "accel": float(accel), "brake": float(brake), "gear": gear}

planner = PlannerSkeleton()

def handle_map_payload(map_payload: Dict[str, Any]) -> None:
    planner.set_map(map_payload)

def planner_step(obs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if not planner.waypoints:
            planner.compute_path(obs)
        return planner.compute_control(obs)
    except Exception as exc:
        print(f"[algo] planner_step error: {exc}")
        return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}