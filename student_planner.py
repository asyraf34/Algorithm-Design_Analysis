"""학생 자율주차 알고리즘 스켈레톤 모듈.

이 파일만 수정하면 되고, 네트워킹/IPC 관련 코드는 `ipc_client.py`에서
자동으로 처리합니다. 학생은 아래 `PlannerSkeleton` 클래스나 `planner_step`
함수를 원하는 로직으로 교체/확장하면 됩니다.
"""

from dataclasses import dataclass
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
import math
from heapq import heappush, heappop


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


def astar(grid: List[List[float]], start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """A* pathfinding with 8-directional movement."""
    H = len(grid)
    W = len(grid[0])

    open_heap: List[Tuple[float, Tuple[int, int]]] = []
    heappush(open_heap, (0.0, start))

    g_cost: Dict[Tuple[int, int], float] = {start: 0.0}
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

    DIR = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < H and 0 <= c < W

    def free(r: int, c: int) -> bool:
        return grid[r][c] == 0

    def h(r: int, c: int) -> float:
        return math.hypot(r - goal[0], c - goal[1])

    while open_heap:
        f, (r, c) = heappop(open_heap)

        if (r, c) == goal:
            path = [(r, c)]
            while (r, c) in parent:
                r, c = parent[(r, c)]
                path.append((r, c))
            return list(reversed(path))

        for dr, dc in DIR:
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc): continue
            if not free(nr, nc): continue

            move_cost = 1.414 if (dr != 0 and dc != 0) else 1.0
            new_g = g_cost[(r, c)] + move_cost

            if new_g < g_cost.get((nr, nc), float("inf")):
                g_cost[(nr, nc)] = new_g
                parent[(nr, nc)] = (r, c)
                f_new = new_g + h(nr, nc)
                heappush(open_heap, (f_new, (nr, nc)))

    return []


@dataclass
class PlannerSkeleton:
    """Two-stage parking planner: align, then enter."""

    map_data: Optional[Dict[str, Any]] = None
    map_extent: Optional[Tuple[float, float, float, float]] = None
    cell_size: float = 0.5
    grid: Optional[List[List[float]]] = None
    plan_grid: Optional[List[List[float]]] = None
    waypoints: List[Tuple[float, float]] = None
    target_orientation: str = "front_in"
    parking_phase: str = "navigate"  # navigate, align, enter, done
    alignment_point: Optional[Tuple[float, float]] = None
    goal_position: Optional[Tuple[float, float]] = None

    def __post_init__(self) -> None:
        if self.waypoints is None:
            self.waypoints = []

    def set_map(self, map_payload: Dict[str, Any]) -> None:
        """Load map and build planning grid."""
        self.map_data = map_payload
        self.map_extent = tuple(
            map(float, map_payload.get("extent", (0.0, 0.0, 0.0, 0.0)))
        )
        self.grid = map_payload["grid"]["stationary"]
        self.cell_size = float(map_payload.get("cellSize", 0.5))
        self.target_orientation = map_payload.get("expected_orientation", "front_in")

        pretty_print_map_summary(map_payload)
        print(f"[algo] Expected orientation: {self.target_orientation}")

        if self.grid is not None:
            raw = self.grid
            H = len(raw)
            W = len(raw[0])

            # Start with stationary obstacles
            inflated = [row[:] for row in raw]

            # Mark occupied parking slots
            slots = map_payload.get("slots", [])
            occupied_idx = map_payload.get("occupied_idx", [])

            for idx, is_occupied in enumerate(occupied_idx):
                if is_occupied and idx < len(slots):
                    slot = slots[idx]
                    xmin_slot, xmax_slot, ymin_slot, ymax_slot = slot

                    try:
                        r1, c1 = self.world_to_grid(xmin_slot, ymin_slot)
                        r2, c2 = self.world_to_grid(xmax_slot, ymax_slot)

                        for r in range(min(r1, r2), max(r1, r2) + 1):
                            for c in range(min(c1, c2), max(c1, c2) + 1):
                                if 0 <= r < H and 0 <= c < W:
                                    inflated[r][c] = 1
                    except:
                        pass

            # Inflate for safety
            temp = [row[:] for row in inflated]
            inflate_radius = 2

            for r in range(H):
                for c in range(W):
                    if temp[r][c] != 0:
                        for dr in range(-inflate_radius, inflate_radius + 1):
                            for dc in range(-inflate_radius, inflate_radius + 1):
                                rr = r + dr
                                cc = c + dc
                                if 0 <= rr < H and 0 <= cc < W:
                                    inflated[rr][cc] = 1

            self.plan_grid = inflated

            blocked = sum(1 for row in self.plan_grid for cell in row if cell != 0)
            total = H * W
            print(f"[algo] Planning grid: {blocked}/{total} cells blocked ({100 * blocked / total:.1f}%)")
        else:
            self.plan_grid = None

        self.waypoints.clear()
        self.parking_phase = "navigate"
        self.alignment_point = None
        self.goal_position = None

    def world_to_grid(self, x, y):
        if self.map_extent is None or self.grid is None:
            raise ValueError("Map not initialized")

        xmin, xmax, ymin, ymax = self.map_extent
        H = len(self.grid)

        col = int((x - xmin) / self.cell_size)
        row_from_bottom = int((y - ymin) / self.cell_size)
        row = H - 1 - row_from_bottom
        return (row, col)

    def grid_to_world(self, r, c):
        if self.map_extent is None or self.grid is None:
            raise ValueError("Map not initialized")

        xmin, xmax, ymin, ymax = self.map_extent

        y = ymax - (r + 0.5) * self.cell_size
        x = xmin + (c + 0.5) * self.cell_size
        return (x, y)

    def find_nearest_free(self, cell: Tuple[int, int]) -> Tuple[int, int]:
        if self.plan_grid is None:
            return cell

        H = len(self.plan_grid)
        W = len(self.plan_grid[0])
        r0, c0 = cell

        q = deque([(r0, c0)])
        visited = {(r0, c0)}

        while q:
            r, c = q.popleft()

            if self.plan_grid[r][c] == 0:
                return (r, c)

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc))

        return cell

    def compute_path(self, obs: Dict[str, Any]) -> None:
        """Compute two-stage path: navigate to alignment point, then enter slot."""
        if self.grid is None:
            print("[algo] No map")
            self.waypoints = []
            return

        s = obs["state"]
        car_x = float(s["x"])
        car_y = float(s["y"])

        ts = obs["target_slot"]
        goal_x = (ts[0] + ts[1]) * 0.5
        goal_y = (ts[2] + ts[3]) * 0.5
        self.goal_position = (goal_x, goal_y)

        print(f"[algo] Car: ({car_x:.1f}, {car_y:.1f}), Goal: ({goal_x:.1f}, {goal_y:.1f})")

        # Calculate alignment point: same Y as goal, offset in X
        slot_width = ts[1] - ts[0]
        slot_height = ts[3] - ts[2]
        is_horizontal = slot_width > slot_height

        if is_horizontal:
            # For horizontal slots: align Y, offset X
            x_offset = 3.5  # meters to the side
            if self.target_orientation == "rear_in":
                # Align on opposite side for backing in
                align_x = goal_x + x_offset if car_x > goal_x else goal_x - x_offset
            else:
                # Align on entry side for front-in
                align_x = goal_x - x_offset if car_x < goal_x else goal_x + x_offset
            align_y = goal_y
        else:
            # For vertical slots: align X, offset Y
            y_offset = 3.5
            align_x = goal_x
            if self.target_orientation == "rear_in":
                align_y = goal_y + y_offset if car_y > goal_y else goal_y - y_offset
            else:
                align_y = goal_y - y_offset if car_y < goal_y else goal_y + y_offset

        self.alignment_point = (align_x, align_y)
        print(f"[algo] Alignment point: ({align_x:.1f}, {align_y:.1f})")

        try:
            start_cell = self.world_to_grid(car_x, car_y)
            align_cell = self.world_to_grid(align_x, align_y)
            goal_cell = self.world_to_grid(goal_x, goal_y)
        except Exception as e:
            print(f"[algo] Grid conversion error: {e}")
            return

        start_cell = self.find_nearest_free(start_cell)
        align_cell = self.find_nearest_free(align_cell)
        goal_cell = self.find_nearest_free(goal_cell)

        print(f"[algo] Grid - Start: {start_cell}, Align: {align_cell}, Goal: {goal_cell}")

        # Path to alignment point
        path_cells = astar(self.plan_grid, start_cell, align_cell)

        if not path_cells:
            print("[algo] No path to alignment point")
            self.waypoints = []
            return

        # Add some intermediate waypoints from alignment to goal
        # This helps the car transition smoothly
        path_cells.append(goal_cell)

        # Convert to world coordinates
        self.waypoints = [self.grid_to_world(r, c) for r, c in path_cells]

        print(f"[algo] Path: {len(self.waypoints)} waypoints")
        print(f"[algo] Waypoints: {self.waypoints}")

    def compute_control(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Two-stage control: navigate to align point, then enter slot."""

        if not self.waypoints:
            self.compute_path(obs)

        if not self.waypoints:
            return {"steer": 0.0, "accel": 0.0, "brake": 0.5, "gear": "D"}

        s = obs["state"]
        x = float(s["x"])
        y = float(s["y"])
        yaw = float(s["yaw"])
        v = float(s["v"])

        ts = obs["target_slot"]
        goal_x = (ts[0] + ts[1]) * 0.5
        goal_y = (ts[2] + ts[3]) * 0.5
        dist_to_goal = math.hypot(goal_x - x, goal_y - y)

        # Update phase based on position
        if self.alignment_point:
            align_x, align_y = self.alignment_point
            dist_to_align = math.hypot(align_x - x, align_y - y)

            if dist_to_goal < 1.0:
                self.parking_phase = "enter"
            elif dist_to_align < 2.0:
                self.parking_phase = "align"
            else:
                self.parking_phase = "navigate"

        print(f"[algo] Phase: {self.parking_phase}, Dist to goal: {dist_to_goal:.2f}")

        # Choose target waypoint
        if self.parking_phase == "enter":
            # Direct to goal
            tx, ty = goal_x, goal_y
            lookahead = 0.5
        elif self.parking_phase == "align":
            # Look ahead slightly
            tx, ty = self.waypoints[min(1, len(self.waypoints) - 1)]
            lookahead = 1.0
        else:
            # Navigate phase: use lookahead
            lookahead = 2.0
            target_idx = 0
            for i, (wx, wy) in enumerate(self.waypoints):
                if math.hypot(wx - x, wy - y) >= lookahead:
                    target_idx = i
                    break
            target_idx = min(target_idx, len(self.waypoints) - 1)
            tx, ty = self.waypoints[target_idx]

        # Prune passed waypoints
        closest_idx = 0
        min_dist = float("inf")
        for i, (wx, wy) in enumerate(self.waypoints):
            d = math.hypot(wx - x, wy - y)
            if d < min_dist:
                min_dist = d
                closest_idx = i

        if closest_idx > 0 and len(self.waypoints) > 2:
            self.waypoints = self.waypoints[closest_idx:]

        # Steering
        dx = tx - x
        dy = ty - y
        target_heading = math.atan2(dy, dx)
        steering = target_heading - yaw
        steering = (steering + math.pi) % (2 * math.pi) - math.pi

        # Smooth steering
        if not hasattr(self, "_prev_steer"):
            self._prev_steer = 0.0
        alpha = 0.6 if self.parking_phase == "enter" else 0.4
        steering = alpha * steering + (1 - alpha) * self._prev_steer
        self._prev_steer = steering

        max_steer = 0.6
        steering = max(-max_steer, min(max_steer, steering))

        # Speed control
        if self.parking_phase == "enter":
            target_speed = 0.4
        elif self.parking_phase == "align":
            target_speed = 0.6
        else:
            if dist_to_goal > 10:
                target_speed = 1.5
            elif dist_to_goal > 6:
                target_speed = 1.0
            else:
                target_speed = 0.7

        # Stop condition
        if dist_to_goal < 0.5 and abs(v) < 0.2:
            return {
                "steer": 0.0,
                "accel": 0.0,
                "brake": 1.0,
                "gear": "D"
            }

        # Gear selection
        gear = "D"
        if self.target_orientation == "rear_in" and self.parking_phase in ["align", "enter"]:
            forward_vec = (math.cos(yaw), math.sin(yaw))
            to_goal = (goal_x - x, goal_y - y)
            goal_dist = math.hypot(*to_goal)
            if goal_dist > 0.1:
                to_goal = (to_goal[0] / goal_dist, to_goal[1] / goal_dist)
                alignment = forward_vec[0] * to_goal[0] + forward_vec[1] * to_goal[1]
                if alignment < -0.2 and dist_to_goal < 4.0:
                    gear = "R"

        # Acceleration
        speed_error = target_speed - abs(v)
        accel_gain = 1.5 if self.parking_phase == "navigate" else 2.0
        accel_cmd = accel_gain * speed_error

        if accel_cmd >= 0.0:
            accel = max(0.0, min(1.0, accel_cmd))
            brake = 0.0
        else:
            accel = 0.0
            brake = max(0.0, min(1.0, -accel_cmd * 1.2))

        return {
            "steer": float(steering),
            "accel": float(accel),
            "brake": float(brake),
            "gear": gear
        }


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

