"""학생 자율주차 알고리즘 - Grid A* + 정밀 진입"""

import heapq
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


@dataclass
class PlannerSkeleton:
    """Grid A* + 정밀 주차 통합"""

    map_data: Optional[Dict[str, Any]] = None
    map_extent: Optional[Tuple[float, float, float, float]] = None
    cell_size: float = 0.5
    stationary_grid: Optional[List[List[float]]] = None
    waypoints: List[Tuple[float, float]] = None
    cached_target: Optional[Tuple[float, float, float, float]] = None
    turn_in_point: Optional[Tuple[float, float]] = None
    turn_in_triggered: bool = False
    goal_axis: Optional[Tuple[float, float]] = None
    parking_active: bool = False
    parking_waypoints: List[Tuple[float, float]] = None
    current_gear: str = "D"
    final_approach_active: bool = False  # ⭐ 최종 진입 플래그

    def __post_init__(self) -> None:
        if self.waypoints is None:
            self.waypoints = []
        if self.parking_waypoints is None:
            self.parking_waypoints = []

    def set_map(self, map_payload: Dict[str, Any]) -> None:
        """맵 설정 - 점유 차량만 장애물"""
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
        self.goal_axis = None
        self.parking_active = False
        self.parking_waypoints.clear()
        self.current_gear = "D"
        self.final_approach_active = False

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

        # 점유 차량만 장애물로
        for idx, is_occ in enumerate(occupied_idx):
            if idx < len(slots) and is_occ:
                mark_rectangle(tuple(slots[idx]))

        # 벽과 선
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

        # 큰 팽창으로 안전 마진 확보
        self.inflate_obstacles(radius_m=1.5)

    def compute_path(self, obs: Dict[str, Any]) -> None:
        """Grid A* 경로 계획 - 맨 위 슬롯 초강력 직진"""
        if not self.map_extent or self.stationary_grid is None:
            return

        slot = obs.get("target_slot") or obs.get("target")
        if slot:
            self.cached_target = tuple(slot)
        elif not self.cached_target:
            return

        self.turn_in_triggered = False
        self.parking_active = False
        self.parking_waypoints.clear()
        self.final_approach_active = False
        self.maneuver_stage = 0
        self.maneuver_triggered = False

        start = (float(obs.get("state", {}).get("x", 0.0)), float(obs.get("state", {}).get("y", 0.0)))
        goal_slot = self.cached_target
        
        min_x, max_x, min_y, max_y = self.map_extent or (0.0, 0.0, 0.0, 0.0)
        
        goal_x_raw = (goal_slot[0] + goal_slot[1]) / 2.0
        goal_y_raw = (goal_slot[2] + goal_slot[3]) / 2.0
        
        margin = 1.0
        goal_x = max(min_x + margin, min(max_x - margin, goal_x_raw))
        goal_y = max(min_y + margin, min(max_y - margin, goal_y_raw))
        
        goal = (goal_x, goal_y)

        span_x = abs(goal_slot[1] - goal_slot[0])
        span_y = abs(goal_slot[3] - goal_slot[2])
        
        # ⭐⭐⭐ 맨 위 슬롯 판별 (y 좌표가 맵 상단)
        is_top_row = goal_y > (max_y - 10.0)
        
        if span_x > span_y:
            goal = (goal[0] - 0.2, goal[1])
            print(f"[algo] horizontal slot, goal shifted left")
        
        axis = (1.0, 0.0) if span_x >= span_y else (0.0, 1.0)
        half_len = max(span_x, span_y) / 2.0
        start_to_goal = (start[0] - goal[0], start[1] - goal[1])
        direction = 1.0 if start_to_goal[0] * axis[0] + start_to_goal[1] * axis[1] >= 0 else -1.0
        self.goal_axis = (axis[0] * direction, axis[1] * direction)

        if span_x >= span_y:
            dist_to_edge = min(abs(goal[0] - min_x), abs(max_x - goal[0]))
        else:
            dist_to_edge = min(abs(goal[1] - min_y), abs(max_y - goal[1]))

        # ⭐⭐⭐ 맨 위 슬롯이면 진입 거리를 훨씬 더 길게
        if is_top_row:
            approach_margin = max(self.cell_size * 6.0, 5.0)  # 5m 이상!
            print(f"[algo] ⭐ TOP ROW - VERY LONG approach (5m+)")
        else:
            approach_margin = max(self.cell_size * 2.0, 1.5)
        
        edge_buffer = max(self.cell_size * 6.0, 3.0)
        if dist_to_edge < edge_buffer:
            approach_margin = max(approach_margin, edge_buffer)
        
        entry_point = (
            goal[0] + axis[0] * direction * (half_len + approach_margin),
            goal[1] + axis[1] * direction * (half_len + approach_margin),
        )
        
        entry_point = (
            max(min_x + margin, min(max_x - margin, entry_point[0])),
            max(min_y + margin, min(max_y - margin, entry_point[1]))
        )
        
        turn_in_offset = max(
            half_len + 0.5 * self.cell_size,
            (half_len + approach_margin) - self.cell_size * 3.0,
        )
        turn_in_point = (
            goal[0] + axis[0] * direction * turn_in_offset,
            goal[1] + axis[1] * direction * turn_in_offset,
        )
        
        turn_in_point = (
            max(min_x + margin, min(max_x - margin, turn_in_point[0])),
            max(min_y + margin, min(max_y - margin, turn_in_point[1]))
        )

        grid_rows = len(self.stationary_grid)
        grid_cols = len(self.stationary_grid[0]) if grid_rows else 0

        if grid_rows == 0 or grid_cols == 0:
            print("[algo] grid empty; using direct goal")
            self.waypoints = [goal]
            return
        
        def world_to_grid(pt: Tuple[float, float]) -> Tuple[int, int]:
            min_x, max_x, min_y, max_y = self.map_extent or (0, 0, 0, 0)
            gx = int((pt[0] - min_x) / self.cell_size)
            gy = int((pt[1] - min_y) / self.cell_size)
            gx = max(0, min(grid_cols - 1, gx))
            gy = max(0, min(grid_rows - 1, gy))
            return gx, gy

        def grid_to_world(idx: Tuple[int, int]) -> Tuple[float, float]:
            min_x, max_x, min_y, max_y = self.map_extent or (0, 0, 0, 0)
            wx = min_x + (idx[0] + 0.5) * self.cell_size
            wy = min_y + (idx[1] + 0.5) * self.cell_size
            return wx, wy

        start_idx = world_to_grid(start)
        goal_idx = world_to_grid(goal)
        entry_idx = world_to_grid(entry_point)
        turn_in_idx = world_to_grid(turn_in_point)

        def nearest_free(idx: Tuple[int, int]) -> Optional[Tuple[int, int]]:
            if not in_bounds(idx):
                return None
            if not is_blocked(idx):
                return idx
            frontier = [idx]
            visited = {idx}
            max_search = 50
            count = 0
            while frontier and count < max_search:
                nxt_frontier: List[Tuple[int, int]] = []
                for cell in frontier:
                    for nb in neighbors(cell):
                        if nb in visited:
                            continue
                        if not in_bounds(nb):
                            continue
                        if not is_blocked(nb):
                            return nb
                        visited.add(nb)
                        nxt_frontier.append(nb)
                frontier = nxt_frontier
                count += 1
            return None

        def in_bounds(idx: Tuple[int, int]) -> bool:
            x, y = idx
            return 0 <= x < grid_cols and 0 <= y < grid_rows

        def is_blocked(idx: Tuple[int, int]) -> bool:
            x, y = idx
            if not in_bounds(idx):
                return True
            return self.stationary_grid[y][x] > 0.5

        def neighbors(idx: Tuple[int, int]) -> Iterable[Tuple[int, int]]:
            x, y = idx
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (x + dx, y + dy)
                if not is_blocked(nxt):
                    yield nxt

        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def a_star(start_pt: Tuple[int, int], goal_pt: Tuple[int, int]) -> List[Tuple[int, int]]:
            frontier: List[Tuple[float, Tuple[int, int]]] = []
            heapq.heappush(frontier, (0.0, start_pt))
            came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start_pt: None}
            cost_so_far: Dict[Tuple[int, int], float] = {start_pt: 0.0}

            while frontier:
                _, current = heapq.heappop(frontier)
                if current == goal_pt:
                    break
                for nxt in neighbors(current):
                    new_cost = cost_so_far[current] + 1.0
                    if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                        cost_so_far[nxt] = new_cost
                        priority = new_cost + heuristic(nxt, goal_pt)
                        heapq.heappush(frontier, (priority, nxt))
                        came_from[nxt] = current

            if goal_pt not in came_from:
                return []

            path: List[Tuple[int, int]] = []
            cur = goal_pt
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return path

        start_free = nearest_free(start_idx)
        entry_free = nearest_free(entry_idx)
        goal_free = nearest_free(goal_idx)
        turn_in_free = nearest_free(turn_in_idx)

        if (
            turn_in_free is not None
            and entry_free is not None
            and turn_in_free == entry_free
        ):
            max_shift_cells = max(1, int(math.ceil(1.0 / self.cell_size)))
            for step in range(1, max_shift_cells + 3):
                candidate_point = (
                    turn_in_point[0] - axis[0] * direction * self.cell_size * step,
                    turn_in_point[1] - axis[1] * direction * self.cell_size * step,
                )
                candidate_free = nearest_free(world_to_grid(candidate_point))
                if candidate_free is not None and candidate_free != entry_free:
                    turn_in_free = candidate_free
                    break

        if turn_in_free is not None:
            self.turn_in_point = grid_to_world(turn_in_free)
        else:
            self.turn_in_point = turn_in_point

        if start_free is None or goal_free is None:
            print(f"[algo] WARNING: no free cell (start={start_idx}, goal={goal_idx})")
            self.waypoints = [goal]
            return

        # 경로 생성
        path_idx: List[Tuple[int, int]] = []
        current_start = start_free
        
        if turn_in_free is not None and turn_in_free != start_free:
            path_to_turn_in = a_star(start_free, turn_in_free)
            if path_to_turn_in:
                path_idx.extend(path_to_turn_in)
                current_start = turn_in_free

        if entry_free is not None and entry_free != current_start:
            path_to_entry = a_star(current_start, entry_free)
            if path_to_entry:
                if path_idx and path_to_entry:
                    path_idx.extend(path_to_entry[1:])
                else:
                    path_idx.extend(path_to_entry)
                current_start = entry_free

        if current_start is not None and goal_free is not None and current_start != goal_free:
            path_from_entry = a_star(current_start, goal_free)
            if path_from_entry:
                path_idx.extend(path_from_entry[1:] if path_idx else path_from_entry)

        if not path_idx and start_free and goal_free:
            path_idx = a_star(start_free, goal_free)
        
        if not path_idx:
            print(f"[algo] A* failed; using direct goal")
            self.waypoints = [goal]
        else:
            self.waypoints = [grid_to_world(p) for p in path_idx]
            
            # ⭐⭐⭐ 맨 위 슬롯이면 슬롯 입구 바로 앞까지 직진 waypoint 추가
            if is_top_row and self.waypoints:
                last_wp = self.waypoints[-1]
                
                # 슬롯 입구 바로 앞 (0.5m 앞)
                front_of_slot = (
                    goal[0] + axis[0] * direction * half_len * 0.5,
                    goal[1] + axis[1] * direction * half_len * 0.5,
                )
                
                # 직진 waypoints 대량 추가
                dist_to_front = math.hypot(front_of_slot[0] - last_wp[0], front_of_slot[1] - last_wp[1])
                num_approach = max(12, int(dist_to_front / 0.3))  # 0.3m마다
                
                for i in range(1, num_approach + 1):
                    ratio = i / num_approach
                    interp_x = last_wp[0] + (front_of_slot[0] - last_wp[0]) * ratio
                    interp_y = last_wp[1] + (front_of_slot[1] - last_wp[1]) * ratio
                    self.waypoints.append((interp_x, interp_y))
                
                print(f"[algo] ⭐ added {num_approach} straight-approach waypoints to FRONT")
            
            # 슬롯 중심까지
            last_wp = self.waypoints[-1]
            dist_to_real_goal = math.hypot(goal[0] - last_wp[0], goal[1] - last_wp[1])
            
            if dist_to_real_goal > 0.3:
                num_steps = max(5, int(dist_to_real_goal / 0.3))
                for i in range(1, num_steps + 1):
                    ratio = i / num_steps
                    interp_x = last_wp[0] + (goal[0] - last_wp[0]) * ratio
                    interp_y = last_wp[1] + (goal[1] - last_wp[1]) * ratio
                    self.waypoints.append((interp_x, interp_y))
                
                print(f"[algo] added {num_steps} fine waypoints to GOAL")
            
            print(f"[algo] ✓ total path len={len(self.waypoints)}")



    def compute_control(self, obs: Dict[str, Any]) -> Dict[str, float]:
        """제어 - 맨 위 슬롯 특별 처리"""
        if not self.parking_active and not self.final_approach_active:
            self.compute_path(obs)

        state = obs.get("state", {})
        x = float(state.get("x", 0.0))
        y = float(state.get("y", 0.0))
        yaw = float(state.get("yaw", 0.0))
        v = float(state.get("v", 0.0))

        limits = obs.get("limits", {})
        max_steer = float(limits.get("maxSteer", 0.6))
        max_accel = float(limits.get("maxAccel", 2.0))
        max_brake = float(limits.get("maxBrake", 5.0))

        cmd = {"steer": 0.0, "accel": 0.0, "brake": 0.0, "gear": self.current_gear}

        goal_slot = self.cached_target
        goal_center = None
        if goal_slot:
            goal_center = (
                (goal_slot[0] + goal_slot[1]) / 2.0,
                (goal_slot[2] + goal_slot[3]) / 2.0,
            )

        # ⭐⭐⭐ 맨 위 슬롯 판별
        min_x, max_x, min_y, max_y = self.map_extent or (0.0, 0.0, 0.0, 0.0)
        is_top_row = goal_center and goal_center[1] > (max_y - 10.0)

        # 거리별 모드 전환
        if goal_slot and goal_center:
            distance_to_goal = math.hypot(goal_center[0] - x, goal_center[1] - y)
            
            # 1.5m 이내: 후진-전진 매뉴버
            if distance_to_goal < 1.5 and not self.maneuver_triggered:
                self.maneuver_triggered = True
                self.final_approach_active = True
                maneuver_seq = self.plan_maneuver_sequence((x, y), goal_slot)
                # (x, y, direction) -> (x, y)로 변환하되 direction 저장
                self.parking_waypoints = [(pt[0], pt[1]) for pt in maneuver_seq]
                self.maneuver_stage = 0
                print(f"[algo] MANEUVER activated: 3-stage (forward-reverse-forward)")
            
            # 3m 이내: 주차 모드
            elif not self.parking_active and not self.final_approach_active and distance_to_goal < 3.0:
                self.parking_active = True
                print(f"[algo] parking mode at {distance_to_goal:.2f}m")

        # Waypoint 정리
        waypoint_threshold = 0.4 if self.maneuver_triggered else 0.5
        
        if self.turn_in_point is not None and not self.turn_in_triggered:
            dist_trigger = math.hypot(self.turn_in_point[0] - x, self.turn_in_point[1] - y)
            if dist_trigger <= max(waypoint_threshold * 1.2, self.cell_size):
                self.turn_in_triggered = True
                print(f"[algo] turn-in triggered")

        active_waypoints = self.parking_waypoints if (self.parking_active or self.final_approach_active) else self.waypoints
        
        while active_waypoints and math.hypot(active_waypoints[0][0] - x, active_waypoints[0][1] - y) < waypoint_threshold:
            popped = active_waypoints.pop(0)
            # ⭐ 매뉴버 단계 진행
            if self.maneuver_triggered and active_waypoints:
                self.maneuver_stage += 1
                print(f"[algo] maneuver stage {self.maneuver_stage}")

        # 목표 도달
        if goal_center:
            final_dist = math.hypot(goal_center[0] - x, goal_center[1] - y)
            if final_dist < 0.3 and abs(v) < 0.05:
                print(f"[algo] ✓ PARKING COMPLETE at {final_dist:.2f}m")
                cmd["brake"] = 1.0
                return cmd

        if not active_waypoints:
            # 재계획
            if self.maneuver_triggered:
                # 매뉴버 완료
                print(f"[algo] maneuver complete")
                cmd["brake"] = 0.5
                return cmd
            
            self.final_approach_active = False
            self.parking_active = False
            self.compute_path(obs)
            active_waypoints = self.waypoints

            if not active_waypoints:
                cmd["brake"] = 0.5
                return cmd

        # Lookahead
        distance_to_goal = math.hypot(active_waypoints[-1][0] - x, active_waypoints[-1][1] - y)
        
        if self.maneuver_triggered:
            lookahead_dist = max(self.cell_size * 0.6, min(self.cell_size * 1.2, distance_to_goal * 0.3))
        elif self.final_approach_active:
            lookahead_dist = max(self.cell_size * 0.8, min(self.cell_size * 1.5, distance_to_goal * 0.4))
        elif self.turn_in_triggered or self.parking_active:
            lookahead_dist = max(self.cell_size, min(self.cell_size * 2.5, distance_to_goal * 0.5))
        else:
            lookahead_dist = max(self.cell_size * 1.2, min(self.cell_size * 3.5, distance_to_goal * 0.6))
        
        target_wp = active_waypoints[0]
        for wp in active_waypoints:
            if math.hypot(wp[0] - x, wp[1] - y) >= lookahead_dist:
                target_wp = wp
                break

        dx = target_wp[0] - x
        dy = target_wp[1] - y
        heading_error = math.atan2(dy, dx) - yaw
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi

        # ⭐⭐⭐ 슬롯 정렬 - 맨 위 슬롯이면 거의 마지막에만
        if is_top_row:
            # 맨 위 슬롯: 0.8m 이내에서만 정렬!
            alignment_distance = 0.8
        else:
            # 다른 슬롯: 2.5m
            alignment_distance = 2.5
        
        if distance_to_goal < alignment_distance and self.goal_axis is not None:
            slot_yaw = math.atan2(self.goal_axis[1], self.goal_axis[0])
            goal_heading_error = slot_yaw - yaw
            while goal_heading_error > math.pi:
                goal_heading_error -= 2 * math.pi
            while goal_heading_error < -math.pi:
                goal_heading_error += 2 * math.pi
            
            if is_top_row:
                # 맨 위 슬롯: 블렌드를 훨씬 더 약하게
                blend = max(0.2, 0.8 - distance_to_goal * 0.5)
            else:
                blend = max(0.4, 1.8 - distance_to_goal * 0.5)
            
            heading_error = 0.5 * heading_error + blend * goal_heading_error  # ⭐ waypoint 비중 증가

        # Pure Pursuit
        wheelbase = float(obs.get("limits", {}).get("L", 2.6))
        curvature = 2.0 * math.sin(heading_error) / max(0.5, math.hypot(dx, dy))
        steer_cmd = math.atan2(wheelbase * curvature, 1.0)
        cmd["steer"] = max(-max_steer, min(max_steer, steer_cmd))

        # ⭐⭐⭐ 속도 제어 (매뉴버 단계별)
        target_speed = 1.5
        
        if self.maneuver_triggered:
            # 매뉴버 중에는 매우 천천히
            if self.maneuver_stage == 1:  # 후진
                target_speed = 0.2
            else:  # 전진
                target_speed = 0.25
        elif self.final_approach_active:
            target_speed = 0.3
        elif self.turn_in_triggered:
            target_speed = min(target_speed, 1.0)
        elif self.parking_active:
            target_speed = min(target_speed, 0.6)
        
        # ⭐⭐⭐ 맨 위 슬롯은 더 천천히 접근
        if is_top_row:
            if distance_to_goal < 4.0:
                target_speed = min(target_speed, 0.8)
            if distance_to_goal < 2.5:
                target_speed = min(target_speed, 0.5)
            if distance_to_goal < 1.5:
                target_speed = min(target_speed, 0.3)
        else:
            if distance_to_goal < 3.0:
                target_speed = min(target_speed, 0.8)
            if distance_to_goal < 1.5:
                target_speed = min(target_speed, 0.5)
            if distance_to_goal < 0.8:
                target_speed = min(target_speed, 0.3)
        
        if abs(heading_error) > 1.0:
            target_speed = min(target_speed, 0.8)

        # ⭐ 기어 결정 (매뉴버 단계별)
        forward_vec = (math.cos(yaw), math.sin(yaw))
        dot_to_target = forward_vec[0] * dx + forward_vec[1] * dy
        
        if self.maneuver_triggered and self.maneuver_stage == 1:
            # 단계 1: 후진
            desired_gear = "R"
        else:
            # 기본: 전진
            desired_gear = "D"

        # 기어 변경
        if desired_gear != self.current_gear and abs(v) > 0.1:
            cmd["accel"] = 0.0
            cmd["brake"] = min(max_brake, 0.9)
        else:
            if desired_gear != self.current_gear:
                self.current_gear = desired_gear
                print(f"[algo] gear changed to {desired_gear}")
            cmd["gear"] = self.current_gear
            
            if v < target_speed:
                cmd["accel"] = min(max_accel, 0.8)
                cmd["brake"] = 0.0
            else:
                cmd["accel"] = 0.0
                cmd["brake"] = min(max_brake, 0.3)

            if distance_to_goal < 0.5:
                cmd["accel"] = 0.0
                cmd["brake"] = min(max_brake, 0.8)

        return cmd

    def plan_parking_sequence(
        self, state: Tuple[float, float], goal_slot: Tuple[float, float, float, float]
    ) -> List[Tuple[float, float]]:
        """최종 정밀 주차 시퀀스"""
        goal = ((goal_slot[0] + goal_slot[1]) / 2.0, (goal_slot[2] + goal_slot[3]) / 2.0)
        span_x = abs(goal_slot[1] - goal_slot[0])
        span_y = abs(goal_slot[3] - goal_slot[2])

        axis = self.goal_axis
        if axis is None:
            axis = (1.0, 0.0) if span_x >= span_y else (0.0, 1.0)

        axis_norm = math.hypot(axis[0], axis[1])
        if axis_norm < 1e-6:
            return [goal]
        axis_unit = (axis[0] / axis_norm, axis[1] / axis_norm)

        half_len = max(span_x, span_y) / 2.0
        
        # ⭐ 슬롯 입구 정렬점
        mouth_point = (
            goal[0] + axis_unit[0] * half_len * 0.7,
            goal[1] + axis_unit[1] * half_len * 0.7,
        )
        
        # 측면 보정
        lateral_dir = (-axis_unit[1], axis_unit[0])
        dx = state[0] - mouth_point[0]
        dy = state[1] - mouth_point[1]
        lateral_error = dx * lateral_dir[0] + dy * lateral_dir[1]
        max_correction = min(span_x, span_y) * 0.2
        correction = max(-max_correction, min(max_correction, lateral_error))
        
        mouth_point = (
            mouth_point[0] + lateral_dir[0] * correction,
            mouth_point[1] + lateral_dir[1] * correction,
        )
        
        sequence = [mouth_point, goal]
        return sequence
    
    def inflate_obstacles(self, radius_m: float = 1.5) -> None:
        """장애물 팽창"""
        if not self.stationary_grid:
            return

        grid_rows = len(self.stationary_grid)
        grid_cols = len(self.stationary_grid[0]) if grid_rows else 0
        radius_cells = max(1, int(math.ceil(radius_m / self.cell_size)))

        inflated = [[0.0 for _ in range(grid_cols)] for _ in range(grid_rows)]
        for y in range(grid_rows):
            for x in range(grid_cols):
                if self.stationary_grid[y][x] <= 0.5:
                    continue
                for dy in range(-radius_cells, radius_cells + 1):
                    for dx in range(-radius_cells, radius_cells + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < grid_cols and 0 <= ny < grid_rows:
                            inflated[ny][nx] = max(inflated[ny][nx], 1.0)

        self.stationary_grid = inflated
        print(f"[algo] inflated obstacles radius={radius_m}m (cells={radius_cells})")


planner = PlannerSkeleton()


def handle_map_payload(map_payload: Dict[str, Any]) -> None:
    planner.set_map(map_payload)


def planner_step(obs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return planner.compute_control(obs)
    except Exception as exc:
        print(f"[algo] error: {exc}")
        return {"steer": 0.0, "accel": 0.0, "brake": 0.5, "gear": "D"}
