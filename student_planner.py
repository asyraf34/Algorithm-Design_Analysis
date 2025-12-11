"""학생 자율주차 알고리즘 스켈레톤 모듈. (수정됨)"""

import heapq
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import time

def normalize_angle(angle: float) -> float:
    while angle > math.pi: angle -= 2.0 * math.pi
    while angle < -math.pi: angle += 2.0 * math.pi
    return angle

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

    def __lt__(self, other):
        return self.f < other.f

@dataclass
class PlannerSkeleton:
    map_data: Optional[Dict[str, Any]] = None
    map_extent: Optional[Tuple[float, float, float, float]] = None
    cell_size: float = 0.5
    stationary_grid: Optional[List[List[float]]] = None
    waypoints: List[Tuple[float, float]] = None
    cached_target: Optional[Tuple[float, float, float, float]] = None
    prev_steer: float = 0.0 
    current_gear_state: int = 1   # 1: Drive, -1: Reverse
    planning_mode: str = "APPROACH" # "APPROACH": Grid A*, "PARKING": Hybrid A*
    last_gear_change_time: float = 0.0 

    def __post_init__(self) -> None:
        if self.waypoints is None:
            self.waypoints = []

    def set_map(self, map_payload: Dict[str, Any]) -> None:
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
        self.expected_orientation = map_payload.get("expected_orientation") or "front_in"

        def mark_rectangle(rect: Tuple[float, float, float, float]) -> None:
            if not self.stationary_grid: return
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

        for wall in map_payload.get("walls_rects", []) or []:
            mark_rectangle(tuple(wall))

        line_thickness = max(self.cell_size * 0.5, 0.2)
        for line in map_payload.get("lines", []) or []:
            if len(line) != 4: continue
            x0, y0, x1, y1 = map(float, line)
            rect = (
                min(x0, x1) - line_thickness, max(x0, x1) + line_thickness,
                min(y0, y1) - line_thickness, max(y0, y1) + line_thickness,
            )
            mark_rectangle(rect)
            
        pretty_print_map_summary(map_payload)
        self.waypoints.clear()

        self.planning_mode = "APPROACH"
        self.inflate_obstacles(0)
    
    def normalize_angle(self, angle: float) -> float:
        while angle > math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle

    def get_grid_index(self, x: float, y: float, yaw: float) -> Tuple[int, int, int]:
        return (int(round(x)), int(round(y)), int(round(yaw / 0.08)))

    def is_collision(self, x: float, y: float) -> bool:
        if not self.stationary_grid: return False
        min_x, max_x, min_y, max_y = self.map_extent or (0, 0, 0, 0)
        
        if not (min_x <= x <= max_x and min_y <= y <= max_y): return True

        gx = int((x - min_x) / self.cell_size)
        gy = int((y - min_y) / self.cell_size)
        
        rows = len(self.stationary_grid)
        cols = len(self.stationary_grid[0]) if rows > 0 else 0
        
        if 0 <= gx < cols and 0 <= gy < rows:
            return self.stationary_grid[gy][gx] > 0.5
        return True

    def inflate_obstacles(self, radius_m: float = 1.0) -> None:
        if not self.stationary_grid: return

        grid_rows = len(self.stationary_grid)
        grid_cols = len(self.stationary_grid[0]) if grid_rows else 0
        radius_cells = max(1, int(math.ceil(radius_m / self.cell_size)))

        inflated = [row[:] for row in self.stationary_grid]
        
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

            if dist >= radius_cells: continue

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid_cols and 0 <= ny < grid_rows:
                    if (nx, ny) not in visited:
                        inflated[ny][nx] = 1.0
                        visited.add((nx, ny))
                        queue.append((nx, ny, dist + 1))

        self.stationary_grid = inflated
        print(f"[algo] inflated obstacles with radius {radius_m}m (cells={radius_cells})")

    def is_obstacle_at(self, x, y):
        if not self.stationary_grid: return True
        min_x, max_x, min_y, max_y = self.map_extent
        if not (min_x <= x <= max_x and min_y <= y <= max_y): return True
        gx = int((x - min_x) / self.cell_size)
        gy = int((y - min_y) / self.cell_size)
        rows = len(self.stationary_grid)
        cols = len(self.stationary_grid[0])
        if 0 <= gx < cols and 0 <= gy < rows:
            return self.stationary_grid[gy][gx] > 0.5
        return True

    def get_final_pose(self, slot_coords):
        x0, x1, y0, y1 = slot_coords
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        width = abs(x1 - x0)
        height = abs(y1 - y0)
        probe_dist = 2.0 
        base_yaw = 0.0
        
        if width > height:
            left_blocked = any(self.is_obstacle_at(x0 - i/40, cy) for i in range(-10, int(probe_dist*40)))
            right_blocked = any(self.is_obstacle_at(x1 + i/40, cy) for i in range(-10, int(probe_dist*40)))
            if left_blocked and not right_blocked: base_yaw = math.pi
            elif not left_blocked and right_blocked: base_yaw = 0.0
            else: base_yaw = 0.0
        else:
            bottom_blocked = any(self.is_obstacle_at(cx, y0 - i/40) for i in range(-10, int(probe_dist*40)))
            top_blocked = any(self.is_obstacle_at(cx, y1 + i/40) for i in range(-10, int(probe_dist*40)))
            if bottom_blocked and not top_blocked: base_yaw = -math.pi / 2.0
            elif not bottom_blocked and top_blocked: base_yaw = math.pi / 2.0
            else: base_yaw = -math.pi / 2.0

        final_yaw = base_yaw
        # if hasattr(self, 'expected_orientation') and self.expected_orientation == 'rear_in':
        #     final_yaw = base_yaw + math.pi
        
        final_yaw = normalize_angle(final_yaw)
        return cx, cy, final_yaw

    def get_entry_pose(self, slot_coords, offset_dist=6.0):
        fx, fy, fyaw = self.get_final_pose(slot_coords)
        target_orientation = getattr(self, 'expected_orientation', 'front_in')
        
        ex = fx - offset_dist * math.cos(fyaw)
        ey = fy - offset_dist * math.sin(fyaw)
        # if target_orientation == 'rear_in':
        #     ex = fx + offset_dist * math.cos(fyaw)
        #     ey = fy + offset_dist * math.sin(fyaw)
        
        return ex, ey

    def a_star_grid(self, start, goal):
        if not self.stationary_grid: return []
        
        min_x, _, min_y, _ = self.map_extent
        def to_grid(val, min_val): return int(round((val - min_val) / self.cell_size))
        def to_world(idx, min_val): return min_val + (idx * self.cell_size)

        sx, sy = to_grid(start[0], min_x), to_grid(start[1], min_y)
        gx, gy = to_grid(goal[0], min_x), to_grid(goal[1], min_y)
        
        rows = len(self.stationary_grid)
        cols = len(self.stationary_grid[0])
        motions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        open_set = []
        heapq.heappush(open_set, (0, sx, sy))
        came_from = {}
        cost_so_far = {(sx, sy): 0}
        
        while open_set:
            _, cx, cy = heapq.heappop(open_set)
            
            if (cx, cy) == (gx, gy): break
            
            for dx, dy in motions:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < cols and 0 <= ny < rows): continue
                if self.stationary_grid[ny][nx] > 0.5: continue
                
                new_cost = cost_so_far[(cx, cy)] + 1
                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    priority = new_cost + abs(gx - nx) + abs(gy - ny)
                    heapq.heappush(open_set, (priority, nx, ny))
                    came_from[(nx, ny)] = (cx, cy)
        
        path = []
        curr = (gx, gy)
        if curr not in came_from: return []
            
        while curr != (sx, sy):
            wx = to_world(curr[0], min_x)
            wy = to_world(curr[1], min_y)
            path.append((wx, wy, 0, 1)) 
            curr = came_from.get(curr)
            if curr is None: break
            
        return path[::-1]

    def hybrid_a_star(self, start, goal):
        import time
        start_node = Node(0, 0, start[0], start[1], start[2], None, self.current_gear_state)
        goal_x, goal_y, goal_yaw = goal

        open_list = []
        heapq.heappush(open_list, start_node)
        
        visited = {}
        visited[self.get_grid_index(start_node.x, start_node.y, start_node.yaw)] = 0.0

        step_size = 0.8
        max_iter = 2000
        wheelbase = 2.5
        
        simulated_max_steer = 0.6*0.92
        steer_actions = [-simulated_max_steer, -simulated_max_steer*2/3, -simulated_max_steer/3, 0, simulated_max_steer/3, simulated_max_steer*2/3, simulated_max_steer]
        
        directions = [1, -1]
        # directions = [1]
        iter_count = 0
        closest_node = start_node
        min_dist_to_goal = float('inf')
        start_time = time.time()
        time_limit = 0.1

        while open_list:
            if time.time() - start_time > time_limit:
                print(f"[algo] Time Limit! Using partial path.")
                break
            if iter_count > max_iter: break
            iter_count += 1

            current = heapq.heappop(open_list)
            dist = math.hypot(current.x - goal_x, current.y - goal_y)
            angle_diff = abs(self.normalize_angle(current.yaw - goal_yaw))
            
            if dist < min_dist_to_goal:
                min_dist_to_goal = dist
                closest_node = current

            # 종료 조건: 거리 0.4m, 각도 ? 이내
            if dist < 0.4 and angle_diff < 0.39:
                closest_node = current
                break

            for d in directions:
                for steer in steer_actions:
                    if abs(steer) < 0.001:
                        next_x = current.x + d * step_size * math.cos(current.yaw)
                        next_y = current.y + d * step_size * math.sin(current.yaw)
                        next_yaw = current.yaw
                    else:
                        if d == -1: continue # 후진과 동시에 턴하는 것이 잘 안 되는 모양
                        turn_radius = wheelbase / math.tan(steer)
                        beta = (d * step_size) / turn_radius
                        cx = current.x - math.sin(current.yaw) * turn_radius
                        cy = current.y + math.cos(current.yaw) * turn_radius
                        next_x = cx + math.sin(current.yaw + beta) * turn_radius
                        next_y = cy - math.cos(current.yaw + beta) * turn_radius
                        next_yaw = self.normalize_angle(current.yaw + beta)

                    if self.is_collision(next_x, next_y): continue
                    dst_nxt_goal = math.hypot(next_x - goal_x, next_y - goal_y)
                    if dst_nxt_goal > 9.2: continue

                    # 비용 함수
                    steer_cost = abs(steer) * 0.2
                    switch_cost = 0.1 if current.direction != d else 0.0
                    rev_cost = 0.1 if d == -1 else 0.0
                    new_g = current.g + step_size + steer_cost + switch_cost + rev_cost
                    
                    # 휴리스틱 강화 (목표 지향적)
                    h_dist = math.hypot(next_x - goal_x, next_y - goal_y)
                    h_angle = abs(self.normalize_angle(next_yaw - goal_yaw))
                    h = (h_dist + h_angle * 2) # 각도 가중치 증가
                    
                    idx = self.get_grid_index(next_x, next_y, next_yaw)
                    if idx not in visited or new_g < visited[idx]:
                        visited[idx] = new_g
                        new_node = Node(new_g + h, new_g, next_x, next_y, next_yaw, current, d)
                        heapq.heappush(open_list, new_node)

        path = []
        node = closest_node
        # if node == closest_node: node = Node(0, 0, goal_x, goal_y, goal_yaw, node, 1)
        while node:
            path.append((node.x, node.y, node.yaw, node.direction))
            node = node.parent
        return path[::-1]

    def compute_path(self, obs: Dict[str, Any]) -> None:
        """
        상태(APPROACH / PARKING)에 따라 경로를 생성합니다.
        + 주차 막바지(1.5m 이내)에는 알고리즘을 끄고 직선 진입을 강제합니다.
        """
        if not self.stationary_grid: return
        slot = obs.get("target_slot") or obs.get("target")
         # [수정] 타겟이 새로 들어오거나 바뀌었으면 그리드 재구축
        if slot:
            new_target = tuple(slot)
            if self.cached_target != new_target:
                self.cached_target = new_target
                self._update_grid_for_target(self.cached_target) # 여기서 벽을 세움
                self.waypoints = [] # 맵이 바뀌었으니 경로 초기화
        if not self.cached_target: return
        
        state = obs.get("state", {})
        sx, sy, syaw = float(state.get("x")), float(state.get("y")), float(state.get("yaw"))

        # 이미 주행 중인 경로가 충분히 남아있으면 재계산 금지
        if self.waypoints and len(self.waypoints) > 0:
            return

        # 목표 위치(주차칸 중심) 계산
        fx, fy, fyaw = self.get_final_pose(self.cached_target)
        
        # 목표까지 남은 거리
        dist_to_final = math.hypot(fx - sx, fy - sy)

        # 1. 접근 모드 (Grid A*)
        if self.planning_mode == "APPROACH":
            ex, ey = self.get_entry_pose(self.cached_target)
            print(f"[algo] Path Planning: APPROACH (Grid A*) -> Entry({ex:.1f}, {ey:.1f})")
            path = self.a_star_grid((sx, sy), (ex, ey))
            
            if path:
                self.waypoints = path
            else:
                print("[algo] Grid A* Failed! Forcing switch to PARKING mode.")
                self.planning_mode = "PARKING" 
        
        # 2. 주차 모드 (Hybrid A*)
        if self.planning_mode == "PARKING":
            
            # [핵심 수정] 주차칸 내부 진입 시(1.5m 이내) 복잡한 계산 금지!
            # Hybrid A*가 각도를 맞추려고 차를 밖으로 빼는 것을 방지함
            if dist_to_final < 1.5:
                print(f"[algo] Final Phase ({dist_to_final:.2f}m): Force Direct Interpolation.")
                
                # 알고리즘을 돌리지 않고, 현재 위치에서 목표 위치로 그냥 선을 그어버림
                # (기어는 현재 기어 유지)
                gear = self.current_gear_state
                
                # 목표점 하나만 딱 찍어줌 (P제어가 알아서 핸들 돌려서 감)
                # 형식: (x, y, yaw, gear)
                self.waypoints = [
                    (sx, sy, syaw, gear), # 시작점(현재위치)
                    (fx, fy, fyaw, gear)  # 끝점(주차칸중심)
                ]
                return

            # 거리가 1.5m 이상일 때만 Hybrid A* 사용
            print(f"[algo] Path Planning: PARKING (Hybrid A*) -> Slot({fx:.1f}, {fy:.1f})")
            path = self.hybrid_a_star((sx, sy, syaw), (fx, fy, fyaw))
            
            if path:
                self.waypoints = path
            else:
                print("[algo] Hybrid A* Failed! Retrying next step...")
                self.waypoints = [] 

    def compute_control(self, obs: Dict[str, Any]) -> Dict[str, float]:
        state = obs.get("state", {})
        x, y = float(state.get("x")), float(state.get("y"))
        yaw = normalize_angle(float(state.get("yaw")))
        v = float(state.get("v", 0.0))

        if not hasattr(self, 'planning_mode'): self.planning_mode = "APPROACH"
        if not hasattr(self, 'current_gear_state'): self.current_gear_state = 1
        
        cmd = {"steer": 0.0, "accel": 0.0, "brake": 0.0, "gear": "D"}

        # 1. 도착 및 모드 전환
        if self.cached_target:
            fx, fy, _ = self.get_final_pose(self.cached_target)
            ex, ey = self.get_entry_pose(self.cached_target)
            dist_to_goal = math.hypot(fx - x, fy - y)
            dist_to_entry = math.hypot(ex - x, ey - y)

            if self.planning_mode == "APPROACH" and dist_to_goal < 8.5 and dist_to_entry < 5.0:
                print(f"[algo] Switch to PARKING mode.")
                self.planning_mode = "PARKING"
                self.waypoints = [] 
                cmd["brake"] = 0.9
                cmd["active_path"] = []
                return cmd

            if self.planning_mode == "PARKING" and dist_to_goal < 0.2:
                print("[algo] Parking Completed.")
                self.waypoints = []
                cmd["brake"] = 1.0
                cmd["active_path"] = []
                return cmd

        if not self.waypoints:
            self.compute_path(obs)
        if not self.waypoints:
            cmd["brake"] = 1.0
            cmd["active_path"] = []
            return cmd

        # 2. 웨이포인트 정리 (일반 주행 중일 때만 삭제)
        while len(self.waypoints) > 1:
            wx, wy, _, w_dir = self.waypoints[0]
            
            # [중요] 다음 점의 기어 방향이 다르면(변곡점), 절대 삭제 금지!
            if w_dir != self.current_gear_state:
                break

            dx = wx - x
            dy = wy - y
            dist = math.hypot(dx, dy)
            local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
            if self.current_gear_state == -1: local_x *= -1 

            # 일반 주행 중 삭제 로직
            should_pop = False
            if dist < 0.5: should_pop = True
            elif local_x < -0.1: should_pop = True
            
            if should_pop:
                self.waypoints.pop(0)
            else:
                break
        
        if not self.waypoints:
            cmd["brake"] = 1.0
            cmd["active_path"] = []
            return cmd

        # ---------------------------------------------------------
        # 3. 타겟 선정 (변곡점 탐색 로직 수정됨)
        # ---------------------------------------------------------
        if self.planning_mode == "PARKING":
            lookahead = 2.0
            steer_gain = 2.0
        else:
            lookahead = 2.0 + 0.3 * abs(v)
            steer_gain = 1.0

        # [핵심 수정] 리스트를 훑어서 "기어가 바뀌는 첫 번째 지점"을 찾음
        next_gear_index = -1
        for i, wp in enumerate(self.waypoints):
            if wp[3] != self.current_gear_state:
                next_gear_index = i
                break
        
        target_wp = None
        dist_to_cusp = float('inf')
        cusp_stop_mode = False

        # 기어 바뀌는 점이 발견되면? -> 그 "직전 점"이 바로 변곡점(Cusp)이다!
        if next_gear_index > 0:
            cusp_index = next_gear_index - 1 # 여기가 멈춰야 할 점
            cusp_wp = self.waypoints[cusp_index]
            dist_to_cusp = math.hypot(cusp_wp[0]-x, cusp_wp[1]-y)
            
            # 변곡점이 시야(5m) 내에 들어오면 -> 변곡점 타겟 락킹
            if dist_to_cusp < 5.0:
                target_wp = cusp_wp
                cusp_stop_mode = True # 정밀 정차 모드 발동
            else:
                # 아직 멀면 그냥 가던 대로 감
                for wp in self.waypoints:
                    if wp[3] != self.current_gear_state: break
                    if math.hypot(wp[0]-x, wp[1]-y) >= lookahead:
                        target_wp = wp
                        break
        
        # 기어 바뀌는 점이 맨 처음(0번)이라면? -> 이미 변곡점을 지났거나 바로 앞임
        elif next_gear_index == 0:
             # 바로 기어 변경 로직으로 넘어가야 함
             cusp_stop_mode = True
             dist_to_cusp = 0.0 # 이미 도달했다고 가정
             target_wp = self.waypoints[0] # 임시
             
        else:
            # 변곡점 없음 (쭉 같은 기어)
            target_wp = self.waypoints[0]
            for wp in self.waypoints:
                if math.hypot(wp[0]-x, wp[1]-y) >= lookahead:
                    target_wp = wp
                    break
        
        if target_wp is None: target_wp = self.waypoints[-1]

        # ---------------------------------------------------------
        # 4. 기어 변경 및 정밀 도달 제어
        # ---------------------------------------------------------
        
        # [핵심 수정] 변곡점 정차 모드일 때
        if cusp_stop_mode:
            # 아직 5cm 이내로 도달 못했으면? -> 계속 밀고 들어가!
            if dist_to_cusp > 0.05:
                # 기어 변경 금지, 현재 기어로 계속 전진
                desired_dir = self.current_gear_state 
                
                # 초근접 시(1m) 기어가듯 천천히
                if dist_to_cusp < 1.0:
                    cmd["accel"] = 0.2 if abs(v) < 0.2 else 0.0 
                    cmd["brake"] = 0.0
                    if abs(v) > 0.3: cmd["brake"] = 0.5 
                    
                    # 조향 계산 후 리턴 (속도 덮어쓰기 방지)
                    dx = target_wp[0] - x
                    dy = target_wp[1] - y
                    target_yaw = math.atan2(dy, dx)
                    
                    # 조향 로직 (기존과 동일)
                    if self.current_gear_state == 1:
                        heading_error = normalize_angle(target_yaw - yaw)
                    else:
                        back_yaw = normalize_angle(yaw + math.pi)
                        heading_error = normalize_angle(target_yaw - back_yaw)
                    
                    raw_steer = steer_gain * heading_error * self.current_gear_state
                    limits = obs.get("limits", {})
                    max_steer = float(limits.get("maxSteer", 0.6))
                    cmd["steer"] = max(-max_steer, min(max_steer, raw_steer))
                    cmd["gear"] = "D" if self.current_gear_state == 1 else "R"
                    cmd["active_path"] = [[wp[0], wp[1], wp[3]] for wp in self.waypoints]
                    return cmd
            
            else:
                # 5cm 이내 도달! (이제 멈추고 기어 바꿀 차례)
                if abs(v) > 0.1: 
                    cmd["brake"] = 1.0
                    cmd["accel"] = 0.0
                    cmd["active_path"] = [[wp[0], wp[1], wp[3]] for wp in self.waypoints]
                    return cmd
                
                # 다음 경로의 기어 방향 가져오기
                next_gear = self.waypoints[next_gear_index][3] if next_gear_index < len(self.waypoints) else self.current_gear_state
                
                print(f"[algo] Cusp Reached! Switch {self.current_gear_state} -> {next_gear}")
                self.current_gear_state = next_gear
                self.last_gear_change_time = time.time()
                
                # [중요] 변곡점(현재 위치)까지 싹 지워버림 -> 남은 건 다음 기어 경로뿐
                if next_gear_index != -1:
                    self.waypoints = self.waypoints[next_gear_index:]
                
                cmd["brake"] = 1.0
                cmd["active_path"] = [[wp[0], wp[1], wp[3]] for wp in self.waypoints]
                return cmd

        # 쿨타임 체크
        if time.time() - self.last_gear_change_time < 1.0:
            pass # 원하는 방향 유지

        cmd["gear"] = "D" if self.current_gear_state == 1 else "R"

        # 5. 조향
        dx = target_wp[0] - x
        dy = target_wp[1] - y
        target_yaw = math.atan2(dy, dx)

        if self.current_gear_state == 1:
            heading_error = normalize_angle(target_yaw - yaw)
        else:
            back_yaw = normalize_angle(yaw + math.pi)
            heading_error = normalize_angle(target_yaw - back_yaw)

        raw_steer = steer_gain * heading_error * self.current_gear_state
        limits = obs.get("limits", {})
        max_steer = float(limits.get("maxSteer", 0.6))
        cmd["steer"] = max(-max_steer, min(max_steer, raw_steer))

        # 6. 속도 제어
        target_speed = 1.5
        
        # 주차 모드 정밀 감속
        if self.planning_mode == "PARKING":
            total_dist = math.hypot(self.waypoints[-1][0]-x, self.waypoints[-1][1]-y)
            if total_dist < 1.5: target_speed = min(target_speed, 0.4)
            if total_dist < 0.5: target_speed = 0.2 # 마지막 진입 시 아주 천천히
            
        elif dist_to_goal > 11:
            target_speed = 5.0
        else:
            target_speed = 3.5

        if self.current_gear_state == -1: target_speed = min(target_speed, 1.0)

        if abs(v) < target_speed:
            cmd["accel"] = 0.5 * (target_speed - abs(v))
        else:
            if abs(v) > target_speed + 0.2: cmd["brake"] = 0.3

        cmd["active_path"] = [[wp[0], wp[1], wp[3]] for wp in self.waypoints]
        return cmd

planner = PlannerSkeleton()

def handle_map_payload(map_payload: Dict[str, Any]) -> None:
    planner.set_map(map_payload)

def planner_step(obs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return planner.compute_control(obs)
    except Exception as exc:
        print(f"[algo] planner_step error: {exc}")
        return {"steer": 0.0, "accel": 0.0, "brake": 0.5, "gear": "D"}