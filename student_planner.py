"""학생 자율주차 알고리즘 스켈레톤 모듈.

이 파일만 수정하면 되고, 네트워킹/IPC 관련 코드는 `ipc_client.py`에서
자동으로 처리합니다. 학생은 아래 `PlannerSkeleton` 클래스나 `planner_step`
함수를 원하는 로직으로 교체/확장하면 됩니다.
"""

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
    """경로 계획/제어 로직을 담는 기본 스켈레톤 클래스입니다."""

    map_data: Optional[Dict[str, Any]] = None
    map_extent: Optional[Tuple[float, float, float, float]] = None
    cell_size: float = 0.5
    stationary_grid: Optional[List[List[float]]] = None
    waypoints: List[Tuple[float, float]] = None
    cached_target: Optional[Tuple[float, float, float, float]] = None

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
        self.inflate_obstacles()

    def compute_path(self, obs: Dict[str, Any]) -> None:
        """관측과 맵을 이용해 경로(웨이포인트)를 준비합니다."""

        if not self.map_extent or self.stationary_grid is None:
            return

        slot = obs.get("target_slot") or obs.get("target")
        if slot:
            self.cached_target = tuple(slot)
        elif not self.cached_target:
            return

        start = (float(obs.get("state", {}).get("x", 0.0)), float(obs.get("state", {}).get("y", 0.0)))
        goal_slot = self.cached_target
        goal = ((goal_slot[0] + goal_slot[1]) / 2.0, (goal_slot[2] + goal_slot[3]) / 2.0)

        # Create an intermediate entry waypoint that sits just outside the parking
        # slot along its longitudinal axis. This encourages the car to begin
        # turning earlier so it approaches the target space with the correct
        # alignment instead of diving in late.
        span_x = abs(goal_slot[1] - goal_slot[0])
        span_y = abs(goal_slot[3] - goal_slot[2])
        axis = (1.0, 0.0) if span_x >= span_y else (0.0, 1.0)
        half_len = max(span_x, span_y) / 2.0
        start_to_goal = (start[0] - goal[0], start[1] - goal[1])
        direction = 1.0 if start_to_goal[0] * axis[0] + start_to_goal[1] * axis[1] >= 0 else -1.0
        approach_margin = max(self.cell_size * 2.0, 1.5)
        entry_point = (
            goal[0] + axis[0] * direction * (half_len + approach_margin),
            goal[1] + axis[1] * direction * (half_len + approach_margin),
        )

        grid_rows = len(self.stationary_grid)
        grid_cols = len(self.stationary_grid[0]) if grid_rows else 0

        if grid_rows == 0 or grid_cols == 0:
            print("[algo] stationary grid empty; falling back to direct goal")
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

        def nearest_free(idx: Tuple[int, int]) -> Optional[Tuple[int, int]]:
            if not in_bounds(idx):
                return None
            if not is_blocked(idx):
                return idx
            # Small BFS search to find the closest free cell if the start/goal lies on an obstacle.
            frontier = [idx]
            visited = {idx}
            while frontier:
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

        if start_free is None or goal_free is None:
            print(
                f"[algo] no free cell for start/goal (start={start_idx}, goal={goal_idx})"
            )
            self.waypoints = [goal]
            return

        # First try to navigate toward the entry point to line up the approach,
        # then proceed into the slot. If the entry cell is blocked, fall back to
        # the direct goal.
        path_idx: List[Tuple[int, int]] = []
        if entry_free is not None:
            path_to_entry = a_star(start_free, entry_free)
            if path_to_entry:
                path_idx.extend(path_to_entry)
                if goal_free != entry_free:
                    path_from_entry = a_star(entry_free, goal_free)
                    if path_from_entry:
                        # Drop the first cell to avoid duplication where the two
                        # paths meet.
                        path_idx.extend(path_from_entry[1:])

        if not path_idx:
            path_idx = a_star(start_free, goal_free)
        if not path_idx:
            print(
                f"[algo] A* failed to find path start={start_free} goal={goal_free}; using direct goal"
            )
            self.waypoints = [goal]
        else:
            self.waypoints = [grid_to_world(p) for p in path_idx]
            print(
                f"[algo] path len={len(path_idx)} start={start_free} goal={goal_free}"
            )

    def compute_control(self, obs: Dict[str, Any]) -> Dict[str, float]:
        """경로를 따라가기 위한 조향/가감속 명령을 산출합니다."""

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

        cmd = {"steer": 0.0, "accel": 0.0, "brake": 0.0, "gear": "D"}

        # Remove visited waypoints
        waypoint_reached_threshold = max(self.cell_size * 0.8, 0.4)
        while self.waypoints and math.hypot(self.waypoints[0][0] - x, self.waypoints[0][1] - y) < waypoint_reached_threshold:
            self.waypoints.pop(0)

        if not self.waypoints:
            cmd["brake"] = 0.5
            return cmd

        # Use a simple lookahead to smooth steering. Selecting a waypoint that is a bit
        # farther ahead helps the vehicle avoid scraping along borders when entering
        # tight parking slots.
        distance_to_goal = math.hypot(self.waypoints[-1][0] - x, self.waypoints[-1][1] - y)
        lookahead_dist = max(self.cell_size * 1.5, min(self.cell_size * 3.0, distance_to_goal * 0.6))
        target_wp = self.waypoints[0]
        for wp in self.waypoints:
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

        steer_gain = 1.2
        cmd["steer"] = max(-max_steer, min(max_steer, steer_gain * heading_error))

        target_speed = 1.5 if distance_to_goal > 3.0 else 0.5

        if v < target_speed:
            cmd["accel"] = min(max_accel, 0.8)
            cmd["brake"] = 0.0
        else:
            cmd["accel"] = 0.0
            cmd["brake"] = min(max_brake, 0.3)

        if distance_to_goal < 0.7:
            cmd["accel"] = 0.0
            cmd["brake"] = min(max_brake, 0.8)

        return cmd

    def inflate_obstacles(self, radius_m: float = 1.5) -> None:
        """단순 팽창으로 차량 폭을 고려한 안전 여유를 확보합니다."""

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