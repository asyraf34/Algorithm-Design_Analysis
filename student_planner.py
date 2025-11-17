"""학생 자율주차 알고리즘 스켈레톤 모듈.

이 파일만 수정하면 되고, 네트워킹/IPC 관련 코드는 `ipc_client.py`에서
자동으로 처리합니다. 학생은 아래 `PlannerSkeleton` 클래스나 `planner_step`
함수를 원하는 로직으로 교체/확장하면 됩니다.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import heapq


def pretty_print_map_summary(map_payload: Dict[str, Any]) -> None:
    extent = map_payload.get("extent") or [None, None, None, None]
    slots = map_payload.get("slots") or []
    occupied = map_payload.get("occupied_idx") or []
    free_slots = len(slots) - sum(1 for v in occupied if v)
    print("[debug] test hybrid A* path planning")
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

    # 차량 파라미터 (필요시 수정)
    wheel_base: float = 2.5
    car_radius: float = 1.5              # 간단한 충돌 체크용 반지름
    max_steer: float = math.radians(35)  # 최대 조향각
    step_distance: float = 0.5           # 한 스텝 이동 거리
    n_theta_bins: int = 72               # 각도 분해 (5도 간격)

    def __post_init__(self) -> None:
        if self.waypoints is None:
            self.waypoints = []

    # ======================
    # 맵 설정
    # ======================
    def set_map(self, map_payload: Dict[str, Any]) -> None:
        """시뮬레이터에서 전송한 정적 맵 데이터를 보관합니다."""
        self.map_data = map_payload
        self.map_extent = tuple(
            map(float, map_payload.get("extent", (0.0, 0.0, 0.0, 0.0)))
        )
        self.cell_size = float(map_payload.get("cellSize", 0.5))
        self.stationary_grid = map_payload.get("grid", {}).get("stationary")
        pretty_print_map_summary(map_payload)
        self.waypoints.clear()

    # ======================
    # 좌표 변환 & 충돌 체크
    # ======================
    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """월드 좌표 (x,y) -> grid index (row, col)"""
        if self.map_extent is None or self.stationary_grid is None:
            return -1, -1
        xmin, ymin, xmax, ymax = self.map_extent
        col = int((x - xmin) / self.cell_size)
        row = int((y - ymin) / self.cell_size)
        return row, col

    def _is_inside_grid(self, x: float, y: float) -> bool:
        if self.stationary_grid is None or self.map_extent is None:
            return False
        rows = len(self.stationary_grid)
        cols = len(self.stationary_grid[0])
        r, c = self._world_to_grid(x, y)
        return 0 <= r < rows and 0 <= c < cols

    def _is_collision(self, x: float, y: float) -> bool:
        """단순 원형 차량 모델로 충돌 체크 (차량 중심 기준)."""
        if self.stationary_grid is None or self.map_extent is None:
            return True

        # 차량 반지름 주변 몇 개 셀만 대략 검사
        r_center, c_center = self._world_to_grid(x, y)
        if r_center < 0:
            return True

        radius_cells = int(self.car_radius / self.cell_size) + 1
        rows = len(self.stationary_grid)
        cols = len(self.stationary_grid[0])

        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                rr = r_center + dr
                cc = c_center + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    if self.stationary_grid[rr][cc] > 0.5:
                        return True
        return False

    # ======================
    # 목표 슬롯 선택 (예시)
    # ======================
    def _select_goal_pose(self) -> Tuple[float, float, float]:
        """
        첫 번째 빈 슬롯의 중심을 goal pose 로 사용.
        실제 슬롯 데이터 포맷에 맞춰 수정 가능.
        """
        if not self.map_data:
            return 0.0, 0.0, 0.0

        slots = self.map_data.get("slots", [])
        occupied = self.map_data.get("occupied_idx", [])
        if not slots:
            return 0.0, 0.0, 0.0

        for i, slot in enumerate(slots):
            occ = occupied[i] if i < len(occupied) else False
            if not occ:
                # 예: slot 이 {'x': ..., 'y': ..., 'yaw': ...} 또는
                # {'pose': {'x':..., 'y':..., 'yaw':...}} 같은 구조라고 가정
                pose = slot.get("pose", slot)
                gx = float(pose.get("x", 0.0))
                gy = float(pose.get("y", 0.0))
                gyaw = float(pose.get("yaw", 0.0))
                return gx, gy, gyaw

        # 모든 슬롯이 찼으면 그냥 맵 중앙을 목표로 (fallback)
        if self.map_extent:
            xmin, ymin, xmax, ymax = self.map_extent
            return (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, 0.0
        return 0.0, 0.0, 0.0

    # ======================
    # Hybrid A* 핵심
    # ======================
    def _normalize_angle(self, angle: float) -> float:
        """[-pi, pi] 범위로 정규화."""
        a = math.fmod(angle + math.pi, 2.0 * math.pi)
        if a < 0:
            a += 2.0 * math.pi
        return a - math.pi

    def _theta_to_index(self, theta: float) -> int:
        th = self._normalize_angle(theta)
        dtheta = 2.0 * math.pi / self.n_theta_bins
        idx = int((th + math.pi) / dtheta)
        if idx < 0:
            idx = 0
        if idx >= self.n_theta_bins:
            idx = self.n_theta_bins - 1
        return idx

    def _heuristic(self, x: float, y: float, gx: float, gy: float) -> float:
        """단순 유클리드 거리 휴리스틱."""
        return math.hypot(gx - x, gy - y)

    def _reconstruct_path(self, came_from_node) -> List[Tuple[float, float]]:
        path = []
        node = came_from_node
        while node is not None:
            path.append((node["x"], node["y"]))
            node = node["parent"]
        path.reverse()
        return path

    def _hybrid_a_star(
        self,
        start: Tuple[float, float, float],
        goal: Tuple[float, float, float],
    ) -> List[Tuple[float, float]]:
        """
        Hybrid A*:
        - 상태: (x, y, yaw)
        - 입력: (전/후진, 조향각)
        """
        if self.stationary_grid is None or self.map_extent is None:
            print("[algo] hybrid_a_star: no map data")
            return []

        sx, sy, syaw = start
        gx, gy, gyaw = goal

        # 시작/목표 충돌 체크
        if self._is_collision(sx, sy):
            print("[algo] start in collision!")
        if self._is_collision(gx, gy):
            print("[algo] goal in collision!")

        # open set (우선순위 큐)
        # 노드 구조: dict 로 관리
        open_heap = []
        start_node = {
            "x": sx,
            "y": sy,
            "yaw": self._normalize_angle(syaw),
            "g": 0.0,
            "h": self._heuristic(sx, sy, gx, gy),
            "parent": None,
            "dir": 1,  # 1: forward, -1: reverse
        }
        heapq.heappush(open_heap, (start_node["g"] + start_node["h"], start_node))

        # 방문 기록: (row, col, theta_idx, dir) -> best_g
        visited = {}

        max_iter = 20000  # 무한 루프 방지용
        iter_count = 0

        # 이동 primitive 정의
        # forward/reverse * steering 3가지 (-max, 0, +max)
        directions = [1, -1]
        steering_list = [-self.max_steer, 0.0, self.max_steer]

        goal_tolerance_pos = 0.8  # m
        goal_tolerance_yaw = math.radians(20.0)

        while open_heap and iter_count < max_iter:
            iter_count += 1
            _, current = heapq.heappop(open_heap)
            cx = current["x"]
            cy = current["y"]
            cyaw = current["yaw"]
            cg = current["g"]

            # goal check
            dist_to_goal = math.hypot(gx - cx, gy - cy)
            yaw_diff = abs(self._normalize_angle(gyaw - cyaw))
            if dist_to_goal < goal_tolerance_pos and yaw_diff < goal_tolerance_yaw:
                print(f"[algo] Hybrid A* goal reached in {iter_count} iters")
                return self._reconstruct_path(current)

            # 방문 체크
            r, c = self._world_to_grid(cx, cy)
            if r < 0:
                continue
            theta_idx = self._theta_to_index(cyaw)
            key = (r, c, theta_idx, current["dir"])
            best_g = visited.get(key, float("inf"))
            if cg >= best_g:
                continue
            visited[key] = cg

            # expand neighbors
            for d in directions:
                for steer in steering_list:
                    # 차량 모델 업데이트
                    ds = self.step_distance * d
                    # bicycle model
                    nx = cx + ds * math.cos(cyaw)
                    ny = cy + ds * math.sin(cyaw)
                    nyaw = cyaw + ds / self.wheel_base * math.tan(steer)
                    nyaw = self._normalize_angle(nyaw)

                    if not self._is_inside_grid(nx, ny):
                        continue
                    if self._is_collision(nx, ny):
                        continue

                    new_g = cg + abs(self.step_distance)
                    new_h = self._heuristic(nx, ny, gx, gy)

                    node = {
                        "x": nx,
                        "y": ny,
                        "yaw": nyaw,
                        "g": new_g,
                        "h": new_h,
                        "parent": current,
                        "dir": d,
                    }
                    heapq.heappush(open_heap, (new_g + new_h, node))

        print("[algo] Hybrid A* failed or hit iteration limit")
        return []

    # ======================
    # public API: 경로 계산
    # ======================
    def compute_path(self, obs: Dict[str, Any]) -> None:
        """관측과 맵을 이용해 경로(웨이포인트)를 준비합니다."""
        self.waypoints.clear()
        
        if self.map_data is None or self.stationary_grid is None:
            print("[algo] compute_path: no map yet")
            print("[debug] test hybrid A* path planning")
            return

        state = obs.get("state", {})
        sx = float(state.get("x", 0.0))
        sy = float(state.get("y", 0.0))
        syaw = float(state.get("yaw", 0.0))

        gx, gy, gyaw = self._select_goal_pose()

        print(f"[algo] start=({sx:.2f},{sy:.2f},{syaw:.2f}), "
              f"goal=({gx:.2f},{gy:.2f},{gyaw:.2f})")

        path = self._hybrid_a_star((sx, sy, syaw), (gx, gy, gyaw))
        if path:
            self.waypoints = path
            print(f"[algo] path length: {len(self.waypoints)}")
        else:
            print("[algo] no path found; waypoints empty")

    # ======================
    # control (임시 그대로)
    # ======================
    def compute_control(self, obs: Dict[str, Any]) -> Dict[str, float]:
        """경로를 따라가기 위한 조향/가감속 명령을 산출합니다.

        지금은 기존 데모 그대로 두고,
        나중에 pure-pursuit 등으로 바꿀 수 있습니다.
        """
        t = float(obs.get("t", 0.0))
        v = float(obs.get("state", {}).get("v", 0.0))

        cmd = {"steer": 0.0, "accel": 0.0, "brake": 0.0, "gear": "D"}

        if t < 2.0:
            cmd["accel"] = 0.6
        elif t < 3.0:
            cmd["brake"] = 0.3
        else:
            cmd["steer"] = 0.07
            if v < 1.0:
                cmd["accel"] = 0.2

        return cmd

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
    
    

    



