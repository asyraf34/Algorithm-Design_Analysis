"""학생 자율주차 알고리즘 스켈레톤 모듈.

이 파일만 수정하면 되고, 네트워킹/IPC 관련 코드는 `ipc_client.py`에서
자동으로 처리합니다. 학생은 아래 `PlannerSkeleton` 클래스나 `planner_step`
함수를 원하는 로직으로 교체/확장하면 됩니다.
"""

import heapq
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import time


def pretty_print_map_summary(map_payload: Dict[str, Any]) -> None:
    extent = map_payload.get("extent") or [None, None, None, None]
    slots = map_payload.get("slots") or []
    occupied = map_payload.get("occupied_idx") or []
    free_slots = len(slots) - sum(1 for v in occupied if v)
    print("[algo] map extent :", extent)
    print("[algo] total slots:", len(slots), "/ free:", free_slots)
    stationary = map_payload.get("grid", {}).get("stationary")
    if stationary:
        rows = len(stationary)
        cols = len(stationary[0]) if stationary else 0
        print("[algo] grid size  :", rows, "x", cols)

class Node:
    def __init__(self, f, g, x, y, yaw, parent=None, direction=1):
        self.f = f
        self.g = g
        self.x = x
        self.y = y
        self.yaw = yaw
        self.parent = parent
        self.direction = direction

    # 힙큐(PriorityQueue)에서 비용(f) 비교를 위해 필요
    def __lt__(self, other):
        return self.f < other.f

@dataclass
class PlannerSkeleton:
    """경로 계획/제어 로직을 담는 기본 스켈레톤 클래스입니다."""

    map_data: Optional[Dict[str, Any]] = None
    map_extent: Optional[Tuple[float, float, float, float]] = None
    cell_size: float = 0.5
    stationary_grid: Optional[List[List[float]]] = None
    waypoints: List[Tuple[float, float]] = None
    cached_target: Optional[Tuple[float, float, float, float]] = None
    turn_in_point: Optional[Tuple[float, float]] = None
    turn_in_triggered: bool = False
    previous_heading_error: Optional[float] = None

    # [추가] 이전 조향각을 기억하기 위한 변수
    prev_steer: float = 0.0 

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
        self.stationary_grid = [list(map(float, row)) for row in raw_grid]
        occupied_idx = map_payload.get("occupied_idx") or []
        slots = map_payload.get("slots") or []
        self.cached_target = None
        self.turn_in_point = None
        self.turn_in_triggered = False

        def mark_rectangle(rect: Tuple[float, float, float, float]) -> None:
            if not self.stationary_grid:
                return

            min_x, max_x, min_y, max_y = self.map_extent or (0, 0, 0, 0)
            x0, x1 = sorted((float(rect[0]), float(rect[1])))
            y0, y1 = sorted((float(rect[2]), float(rect[3])))
            start_x = max(0, int((x0 - min_x) / self.cell_size))
            end_x = min(len(self.stationary_grid[0]) - 1, int(math.ceil((x1 - min_x) / self.cell_size)))
            start_y = max(0, int((y0 - min_y) / self.cell_size))
            end_y = min(len(self.stationary_grid) - 1, int(math.ceil((y1 - min_y) / self.cell_size)))
            for gy in range(start_y, end_y + 1):
                for gx in range(start_x, end_x + 1):
                    self.stationary_grid[gy][gx] = max(self.stationary_grid[gy][gx], 1.0)

        for idx, is_occ in enumerate(occupied_idx):
            if idx < len(slots) and is_occ:
                mark_rectangle(tuple(slots[idx]))

        # Treat map borders and painted lines as hard obstacles so that the planner
        # never clips through them. Some maps only encode these as rectangles/lines
        # outside the stationary grid, so we explicitly rasterize them into the
        # occupancy grid here.
        for wall in map_payload.get("walls_rects", []) or []:
            mark_rectangle(tuple(wall))

        line_thickness = max(self.cell_size * 0.5, 0.2)
        for line in map_payload.get("lines", []) or []:
            if len(line) != 4:
                continue
            x0, y0, x1, y1 = map(float, line)
            rect = (
                min(x0, x1) - line_thickness,
                max(x0, x1) + line_thickness,
                min(y0, y1) - line_thickness,
                max(y0, y1) + line_thickness,
            )
            mark_rectangle(rect)
            
        pretty_print_map_summary(map_payload)
        self.waypoints.clear()

        # Inflate obstacles to give the ego vehicle clearance against borders/parked cars.
        # This helps avoid collisions caused by planning too close to obstacles.
        self.inflate_obstacles(0.3)
    
    # 기존 PlannerSkeleton 내부의 메서드들을 이것들로 교체하세요

    def normalize_angle(self, angle: float) -> float:
        while angle > math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle

    def get_grid_index(self, x: float, y: float, yaw: float) -> Tuple[int, int, int]:
        # 1.0m 단위 그리드, 15도(0.26rad) 단위 각도 (속도를 위해 그리드를 키움)
        return (int(round(x)), int(round(y)), int(round(yaw / 0.26)))

    def is_collision(self, x: float, y: float) -> bool:
        if not self.stationary_grid: return False
        min_x, max_x, min_y, max_y = self.map_extent or (0, 0, 0, 0)
        
        # 맵 밖으로 나가면 충돌
        if not (min_x <= x <= max_x and min_y <= y <= max_y): return True

        # 그리드 인덱스 변환
        gx = int((x - min_x) / self.cell_size)
        gy = int((y - min_y) / self.cell_size)
        
        rows = len(self.stationary_grid)
        cols = len(self.stationary_grid[0]) if rows > 0 else 0
        
        if 0 <= gx < cols and 0 <= gy < rows:
            # 0.5 이상이면 장애물로 간주
            return self.stationary_grid[gy][gx] > 0.5
        return True

    def hybrid_a_star(self, start, goal):
        """
        """
        import time # 함수 내 import 혹은 파일 상단 이동
        
        if self.is_collision(start[0], start[1]):
            print(f"[algo] FAIL: Start {start} is in OBSTACLE!")
            return []
        
        # Node: (f, g, x, y, yaw, parent, direction)
        start_node = Node(0, 0, start[0], start[1], start[2], None, 1)
        goal_x, goal_y, goal_yaw = goal

        open_list = []
        heapq.heappush(open_list, start_node)
        
        visited = {}
        visited[self.get_grid_index(start_node.x, start_node.y, start_node.yaw)] = 0.0

        # 파라미터 설정
        step_size = 1.0
        max_iter = 5000
        wheelbase = 2.5
        max_steer = 0.6
        steer_actions = [-max_steer, 0, max_steer]
        directions = [1, -1] # 1: 전진, -1: 후진

        iter_count = 0
        
        # 타임아웃 및 최적해 보존용
        closest_node = start_node
        min_dist_to_goal = float('inf')
        start_time = time.time()
        time_limit = 0.15 

        while open_list:
            if time.time() - start_time > time_limit:
                print(f"[algo] Time Limit! ({time_limit}s) Using best partial path.")
                break

            if iter_count > max_iter:
                print("[algo] Max iterations reached. Using best partial path.")
                break
            iter_count += 1

            current = heapq.heappop(open_list)

            dist = math.hypot(current.x - goal_x, current.y - goal_y)
            angle_diff = abs(self.normalize_angle(current.yaw - goal_yaw))
            
            # 가장 가까운 노드 기록
            if dist < min_dist_to_goal:
                min_dist_to_goal = dist
                closest_node = current

            # 종료 조건 (거리 1.0m, 각도 35도 이내)
            if dist < 1.0 and (angle_diff < 0.6 or abs(angle_diff - math.pi) < 0.6):
                closest_node = current
                break

            for d in directions:
                for steer in steer_actions:
                    # 이동 모델
                    if abs(steer) < 0.001:
                        next_x = current.x + d * step_size * math.cos(current.yaw)
                        next_y = current.y + d * step_size * math.sin(current.yaw)
                        next_yaw = current.yaw
                    else:
                        turn_radius = wheelbase / math.tan(steer)
                        beta = (d * step_size) / turn_radius
                        cx = current.x - math.sin(current.yaw) * turn_radius
                        cy = current.y + math.cos(current.yaw) * turn_radius
                        next_x = cx + math.sin(current.yaw + beta) * turn_radius
                        next_y = cy - math.cos(current.yaw + beta) * turn_radius
                        next_yaw = self.normalize_angle(current.yaw + beta)

                    if self.is_collision(next_x, next_y): continue

                    # ---------------------------------------------------------
                    # [핵심 수정] 비용 함수
                    # ---------------------------------------------------------
                    steer_cost = abs(steer) * 0.1
                    
                    # 기어 변경 비용 (매우 비싸게 -> 한 번 움직이면 쭉 가게 함)
                    switch_cost = 30.0 if current.direction != d else 0.0
                    
                    # 후진 비용 (매우 싸게 -> 후진 두려워하지 않음)
                    rev_cost = 0.1 if d == -1 else 0.0
                    
                    new_g = current.g + step_size + steer_cost + switch_cost + rev_cost
                    
                    # 휴리스틱 (가중치 1.5배)
                    h_dist = math.hypot(next_x - goal_x, next_y - goal_y)
                    h_angle = abs(self.normalize_angle(next_yaw - goal_yaw))
                    h = (h_dist + h_angle * 2.0) * 1.5 
                    
                    idx = self.get_grid_index(next_x, next_y, next_yaw)
                    
                    if idx not in visited or new_g < visited[idx]:
                        visited[idx] = new_g
                        # 방향(d) 정보 저장
                        new_node = Node(new_g + h, new_g, next_x, next_y, next_yaw, current, d)
                        heapq.heappush(open_list, new_node)

        # 경로 복원 (x, y, direction)
        path = []
        node = closest_node
        while node:
            path.append((node.x, node.y, node.direction))
            node = node.parent
        
        # print(f"[algo] Iter: {iter_count}, Time: {time.time()-start_time:.3f}s")
        return path[::-1]

    def compute_path(self, obs: Dict[str, Any]) -> None:
        if not self.map_extent or self.stationary_grid is None: return

        slot = obs.get("target_slot") or obs.get("target")
        if slot: self.cached_target = tuple(slot)
        if not self.cached_target: return

        state = obs.get("state", {})
        sx, sy, syaw = float(state.get("x", 0)), float(state.get("y", 0)), float(state.get("yaw", 0))
        
        goal_slot = self.cached_target
        gx = (goal_slot[0] + goal_slot[1]) / 2.0
        gy = (goal_slot[2] + goal_slot[3]) / 2.0

        dist_to_goal = math.hypot(gx - sx, gy - sy)

        # [수정된 Lock 로직]
        # 3.0m 이내이고 "경로가 남아있을 때만" 재계산 금지.
        # 즉, compute_control에서 경로를 날려버렸으면(len=0), 여기서 뚫고 지나가서 재계산합니다.
        if dist_to_goal < 3.0 and self.waypoints and len(self.waypoints) > 0:
            return

        if self.waypoints and len(self.waypoints) > 5:
            return
        
        span_x = abs(goal_slot[1] - goal_slot[0])
        span_y = abs(goal_slot[3] - goal_slot[2])
        gyaw = 0.0 if span_x > span_y else (math.pi / 2.0)

        print(f"[algo] Planning Start.. {sx:.1f},{sy:.1f} -> {gx:.1f},{gy:.1f}")

        path = self.hybrid_a_star((sx, sy, syaw), (gx, gy, gyaw))
        
        if path:
            self.waypoints = path
            self.previous_heading_error = None
            print(f"[algo] Path Found! Length: {len(path)}")
        else:
            print("[algo] A* Failed. Using fallback direct path.")
            self.waypoints = [
                (sx, sy, 1),      # 시작점 (현재 내 위치)
                (gx, gy, 1)       # 목표점
            ]

    def compute_control(self, obs: Dict[str, Any]) -> Dict[str, float]:
        state = obs.get("state", {})
        x, y = float(state.get("x")), float(state.get("y"))
        yaw = self.normalize_angle(float(state.get("yaw")))
        v = float(state.get("v", 0.0))

        if not hasattr(self, 'current_gear_state'):
            self.current_gear_state = 1 

        # 0. 도착 판정 (최우선)
        if self.cached_target:
            goal_slot = self.cached_target
            gx = (goal_slot[0] + goal_slot[1]) / 2.0
            gy = (goal_slot[2] + goal_slot[3]) / 2.0
            if math.hypot(gx - x, gy - y) < 0.5:
                print("[algo] [Stop Reason] Goal Reached (Dist < 0.5m)")
                self.waypoints = [] 
                return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}

        self.compute_path(obs)

        limits = obs.get("limits", {})
        max_steer = float(limits.get("maxSteer", 0.6))
        cmd = {"steer": 0.0, "accel": 0.0, "brake": 0.0, "gear": "D"}

        # ==================================================================
        # [핵심 수정] 진행 방향 뒤쪽 웨이포인트 삭제 (Dot Product)
        # ==================================================================
        while self.waypoints:
            wp_x, wp_y = self.waypoints[0][:2]
            dx = wp_x - x
            dy = wp_y - y
            dist = math.hypot(dx, dy)

            # 내적(Dot Product) 계산: 내 차의 앞뒤 방향 기준 위치 (local_x)
            # 양수(+)면 내 차 앞, 음수(-)면 내 차 뒤
            local_x = dx * math.cos(yaw) + dy * math.sin(yaw)

            # 마지막 점은 함부로 지우지 않음 (0.5m까지 접근 허용)
            is_last_point = (len(self.waypoints) == 1)
            
            should_pop = False

            if is_last_point:
                if dist < 0.5: should_pop = True
            else:
                # 1. 너무 가까우면 삭제 (기존 로직)
                if dist < 1.0: 
                    should_pop = True
                
                # 2. [추가된 로직] 진행 방향보다 뒤에 있으면 삭제
                # 거리 2.5m 이내일 때만 적용 (너무 먼 점을 지우지 않도록)
                elif dist < 2.5:
                    if self.current_gear_state == 1: # [전진 중]
                        # 점이 내 차 뒤(-0.1m)로 넘어갔다면 삭제
                        if local_x < -0.1: should_pop = True
                    else: # [후진 중]
                        # 후진 중에는 내 차 '앞(양수)'에 있는 점이 실제로는 뒤로 지나친 점임
                        if local_x > 0.1: should_pop = True

            if should_pop:
                self.waypoints.pop(0)
            else:
                break
        # ==================================================================

        if not self.waypoints:
            # 경로가 없으면 정지
            cmd["brake"] = 1.0
            return cmd

        # ... (이하 타겟 설정, 기어 변경, 주행 제어 등 기존 로직 유지) ...
        # (아래 코드는 이전에 드린 코드와 동일합니다. 연결을 위해 생략하지 않고 적어드립니다)

        # 2. 경로 이탈 감지 (완화됨)
        if self.waypoints:
            dists = [math.hypot(wp[0] - x, wp[1] - y) for wp in self.waypoints]
            min_dist_to_path = min(dists)
            if min_dist_to_path > 2.0 and abs(v) > 0.5:
                print(f"[algo] [Stop Reason] Path Deviation ({min_dist_to_path:.2f}m). Coasting.")
                self.waypoints = [] 
                cmd["accel"] = 0.0
                cmd["brake"] = 0.0 
                return cmd

        # 3. 타겟 웨이포인트 설정
        target_wp = self.waypoints[0]
        lookahead_dist = 2.0
        for wp in self.waypoints:
            if math.hypot(wp[0] - x, wp[1] - y) >= lookahead_dist:
                target_wp = wp
                break
        
        # 4. 기어 변경 로직
        desired_dir = target_wp[2] if len(target_wp) > 2 else 1
        
        if desired_dir != self.current_gear_state:
            if abs(v) > 0.05:
                cmd["gear"] = "D" if self.current_gear_state == 1 else "R"
                cmd["accel"] = 0.0
                cmd["brake"] = 1.0
                return cmd 
            else:
                self.current_gear_state = desired_dir
        
        # 5. 주행 제어
        min_accel = 0.4 if abs(v) < 0.3 else 0.0

        if self.current_gear_state == 1: # [전진]
            cmd["gear"] = "D"
            dx = target_wp[0] - x
            dy = target_wp[1] - y
            target_yaw = math.atan2(dy, dx)
            heading_error = self.normalize_angle(target_yaw - yaw)
            
            cmd["steer"] = 0.8 * heading_error

            target_speed = 1.5 if len(self.waypoints) > 2 else 0.5
            if v < target_speed:
                cmd["accel"] = max(min_accel, 0.5 * (target_speed - v))
                cmd["brake"] = 0.0
            else:
                cmd["accel"] = 0.0
                if v > target_speed + 0.3: cmd["brake"] = 0.3

        else: # [후진]
            cmd["gear"] = "R"
            dx = target_wp[0] - x
            dy = target_wp[1] - y
            target_yaw = math.atan2(dy, dx)
            
            back_yaw = self.normalize_angle(yaw + math.pi)
            heading_error = self.normalize_angle(target_yaw - back_yaw)
            
            cmd["steer"] = 0.6 * heading_error 
            
            target_speed_rev = 1.0
            current_speed_abs = abs(v)
            
            if current_speed_abs < target_speed_rev:
                calc_accel = 0.4 * (target_speed_rev - current_speed_abs)
                cmd["accel"] = max(min_accel, calc_accel)
                cmd["brake"] = 0.0
            else:
                cmd["accel"] = 0.0
                if current_speed_abs > target_speed_rev + 0.3:
                    cmd["brake"] = 0.3

        cmd["steer"] = max(-max_steer, min(max_steer, cmd["steer"]))
        alpha = 0.2
        if self.prev_steer is None: self.prev_steer = 0.0
        smoothed_steer = alpha * self.prev_steer + (1 - alpha) * cmd["steer"]
        cmd["steer"] = smoothed_steer
        self.prev_steer = smoothed_steer
        
        return cmd

    def inflate_obstacles(self, radius_m: float = 1.5) -> None:
        """단순 팽창으로 차량 폭을 고려한 안전 여유를 확보합니다. (BFS 최적화 버전)"""

        if not self.stationary_grid:
            return

        grid_rows = len(self.stationary_grid)
        grid_cols = len(self.stationary_grid[0]) if grid_rows else 0
        radius_cells = max(1, int(math.ceil(radius_m / self.cell_size)))

        inflated = [row[:] for row in self.stationary_grid]  # 복사
        
        # 큐 초기화: 모든 장애물 셀을 (x, y, dist) 형태로 추가
        queue = []
        for r in range(grid_rows):
            for c in range(grid_cols):
                if inflated[r][c] > 0.5:
                    queue.append((c, r, 0))

        visited = set((q[0], q[1]) for q in queue)

        head = 0
        while head < len(queue):
            x, y, dist = queue[head]
            head += 1

            if dist >= radius_cells:
                continue

            # 4방향 또는 8방향 이웃 탐색
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = x + dx, y + dy

                if 0 <= nx < grid_cols and 0 <= ny < grid_rows:
                    if (nx, ny) not in visited:
                        inflated[ny][nx] = 1.0
                        visited.add((nx, ny))
                        queue.append((nx, ny, dist + 1))

        self.stationary_grid = inflated
        print(
            f"[algo] inflated obstacles with radius {radius_m}m (cells={radius_cells})"
        )

# 전역 planner 인스턴스 (통신 모듈이 이 객체를 사용합니다.)
planner = PlannerSkeleton()


def handle_map_payload(map_payload: Dict[str, Any]) -> None:
    """통신 모듈에서 맵 패킷을 받을 때 호출됩니다."""

    planner.set_map(map_payload)


def planner_step(obs: Dict[str, Any]) -> Dict[str, Any]:
    """통신 모듈에서 매 스텝 호출하여 명령을 생성합니다."""

    try:
        return planner.compute_control(obs)
    except Exception as exc:
        print(f"[algo] planner_step error: {exc}")
        return {"steer": 0.0, "accel": 0.0, "brake": 0.5, "gear": "D"}