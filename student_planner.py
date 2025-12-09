"""학생 자율주차 알고리즘 스켈레톤 모듈.

이 파일만 수정하면 되고, 네트워킹/IPC 관련 코드는 `ipc_client.py`에서
자동으로 처리합니다. 학생은 아래 `PlannerSkeleton` 클래스나 `planner_step`
함수를 원하는 로직으로 교체/확장하면 됩니다.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math


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


def _normalize_angle(angle: float) -> float:
    """각도를 [-pi, pi] 범위로 정규화."""
    a = math.fmod(angle + math.pi, 2.0 * math.pi)
    if a < 0:
        a += 2.0 * math.pi
    return a - math.pi


@dataclass
class PlannerSkeleton:
    """경로 계획/제어 로직을 담는 기본 스켈레톤 클래스입니다."""

    map_data: Optional[Dict[str, Any]] = None
    map_extent: Optional[Tuple[float, float, float, float]] = None
    cell_size: float = 0.5
    stationary_grid: Optional[List[List[float]]] = None

    waypoints: List[Tuple[float, float]] = None
    straight_in_mode: bool = False
    target_center: Optional[Tuple[float, float]] = None
    target_slot_idx: Optional[int] = None
    lane_y: Optional[float] = None
    path_planned: bool = False
    slot_forward_heading: float = 0.0   # 슬롯 안으로 들어가는 방향 (yaw)
    slot_depth_half: float = 2.5        # 슬롯 깊이의 절반

    def __post_init__(self) -> None:
        if self.waypoints is None:
            self.waypoints = []

    # ===================== 맵 수신 =====================

    def set_map(self, map_payload: Optional[Dict[str, Any]]) -> None:
        """시뮬레이터에서 전송한 정적 맵 데이터를 보관합니다."""
        if map_payload is None:
            print("[algo] set_map: got None map_payload, ignoring")
            return

        self.map_data = map_payload
        self.map_extent = tuple(
            map(float, map_payload.get("extent", (0.0, 0.0, 0.0, 0.0)))
        )
        self.cell_size = float(map_payload.get("cellSize", 0.5))
        self.stationary_grid = map_payload.get("grid", {}).get("stationary")
        pretty_print_map_summary(map_payload)

        # 맵이 바뀌면 상태 초기화
        self.waypoints.clear()
        self.straight_in_mode = False
        self.target_center = None
        self.target_slot_idx = None
        self.lane_y = None
        self.path_planned = False

    # ===================== target_slot 처리 =====================

    def _extract_target_center(self, obs: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """obs['target_slot']에서 중심 좌표 (cx, cy)를 얻는다."""
        ts = obs.get("target_slot")
        if ts is None:
            return None

        # dict 형태: {"xmin":..,"xmax":..,"ymin":..,"ymax":..}
        if isinstance(ts, dict):
            xmin = ts.get("xmin")
            xmax = ts.get("xmax")
            ymin = ts.get("ymin")
            ymax = ts.get("ymax")

        # 리스트/튜플: [xmin, xmax, ymin, ymax] 또는 [[xmin, ymin],[xmax, ymax]]
        elif isinstance(ts, (list, tuple)):
            if len(ts) == 4 and all(isinstance(v, (int, float)) for v in ts):
                xmin, xmax, ymin, ymax = ts
            elif len(ts) == 2 and all(isinstance(v, (list, tuple)) for v in ts):
                (xmin, ymin), (xmax, ymax) = ts
            else:
                return None
        else:
            return None

        try:
            cx = (float(xmin) + float(xmax)) * 0.5
            cy = (float(ymin) + float(ymax)) * 0.5
            return cx, cy
        except Exception:
            return None

    def _compute_lane_y(self, slot_center: Tuple[float, float]) -> Optional[float]:
        """슬롯이 밑/가운데/위 줄 중 어디에 있는지에 따라 lane_y를 정한다."""
        if self.map_extent is None:
            return None
        xmin, xmax, ymin, ymax = self.map_extent
        if ymin is None or ymax is None:
            return None

        cx, cy = slot_center

        height = ymax - ymin
        band_h = height / 3.0

        # y 기준으로 줄 구분
        if cy < ymin + band_h:
            band = "bottom"   # 맨 아래 줄
        elif cy < ymin + 2 * band_h:
            band = "middle"   # 가운데 줄
        else:
            band = "top"      # 맨 위 줄

        offset = 8.0   # 슬롯 줄에서 얼마나 떨어질지
        margin = 2.0   # 벽에서 최소 거리

        if band == "top":
            # 윗줄이면 아래쪽 차선에서 올라가도록
            lane_y = max(cy - offset, ymin + margin)
            side = "down"
        else:
            # 밑줄/가운데줄은 위쪽 차선에서 내려오도록
            lane_y = min(cy + offset, ymax - margin)
            side = "up"

        print(
            f"[algo] band={band}, slot_center_y={cy:.2f} -> "
            f"lane_y={lane_y:.2f} (from {side})"
        )
        return lane_y

    def _find_target_slot_index(self, center: Tuple[float, float]) -> Optional[int]:
        """map_data['slots'] 중 target_center와 가장 가까운 슬롯 인덱스를 찾는다.

        slots 원소 형식:
        - dict: {"xmin":..,"xmax":..,"ymin":..,"ymax":..}
        - list/tuple: [xmin, xmax, ymin, ymax]
        - list/tuple: [[xmin, ymin], [xmax, ymax]]
        """
        if not self.map_data:
            return None

        slots = self.map_data.get("slots") or []
        if not slots:
            return None

        cx, cy = center
        best_i: Optional[int] = None
        best_d2 = float("inf")

        for i, s in enumerate(slots):
            try:
                # dict 형식
                if isinstance(s, dict):
                    sx = (float(s["xmin"]) + float(s["xmax"])) * 0.5
                    sy = (float(s["ymin"]) + float(s["ymax"])) * 0.5

                # list/tuple 형식
                elif isinstance(s, (list, tuple)):
                    # [xmin, xmax, ymin, ymax]
                    if len(s) == 4 and all(isinstance(v, (int, float)) for v in s):
                        sx = (float(s[0]) + float(s[1])) * 0.5
                        sy = (float(s[2]) + float(s[3])) * 0.5
                    # [[xmin, ymin], [xmax, ymax]]
                    elif len(s) == 2 and all(isinstance(v, (list, tuple)) for v in s):
                        (xmin, ymin2), (xmax, ymax2) = s
                        sx = (float(xmin) + float(xmax)) * 0.5
                        sy = (float(ymin2) + float(ymax2)) * 0.5
                    else:
                        continue
                else:
                    continue

            except Exception:
                continue

            d2 = (sx - cx) ** 2 + (sy - cy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i

        self.target_slot_idx = best_i
        print(f"[algo] target_slot_idx = {best_i}")
        return best_i

    # ===================== 특수 경로 (윗줄 1~3 슬롯) =====================

    def _compute_special_top_left_path(self, obs: Dict[str, Any], center: Tuple[float, float]) -> None:
        """윗줄 1,2,3번 슬롯(인덱스 0~2) 전용 안전 경로 생성."""
        if self.map_extent is None:
            return

        cx, cy = center
        xmin, xmax, ymin, ymax = self.map_extent

        # 슬롯 폭 추정
        slot_width = 5.0
        ts = obs.get("target_slot")
        try:
            if isinstance(ts, dict):
                slot_width = abs(float(ts["xmax"]) - float(ts["xmin"]))
            elif isinstance(ts, (list, tuple)) and len(ts) == 4:
                slot_width = abs(float(ts[1]) - float(ts[0]))
        except Exception:
            pass

        # 현재 상태
        state = obs.get("state", {})
        x = float(state.get("x", 0.0))
        y = float(state.get("y", 0.0))
        yaw = float(state.get("yaw", 0.0))

        waypoints: List[Tuple[float, float]] = []

        # 1) 왼쪽 세로 통로를 따라 안쪽으로 충분히 내려오기
        inner_y = ymin + (ymax - ymin) * 0.45   # 지도 높이의 약 중간
        if abs(y - inner_y) > 1.0:
            waypoints.append((x, inner_y))

        # 2) 맵 안쪽으로 x 방향 이동 (왼쪽 벽에서 멀어지기)
        safe_x = xmin + slot_width * 3.5
        if safe_x < x + 2.0:
            safe_x = x + 2.0
        waypoints.append((safe_x, inner_y))

        # 3) 위쪽 줄 아래쪽 차선으로 이동
        offset = 4.5
        lane_y = cy - offset
        if lane_y < ymin + 2.0:
            lane_y = ymin + 2.0
        waypoints.append((safe_x, lane_y))

        # 4) 슬롯 x 위치로 접근
        approach_offset = 3.5
        approach_y = cy - approach_offset
        waypoints.append((cx, approach_y))

        # 5) 최종 정지 위치
        stop_ratio = 0.55
        stop_y = approach_y + stop_ratio * (cy - approach_y)
        waypoints.append((cx, stop_y))

        self.waypoints = waypoints
        self.lane_y = lane_y
        self.straight_in_mode = False
        self.path_planned = True

        # 슬롯 깊이 및 진입 방향 설정
        depth = 5.0
        try:
            if isinstance(ts, dict):
                dx = abs(float(ts["xmax"]) - float(ts["xmin"]))
                dy = abs(float(ts["ymax"]) - float(ts["ymin"]))
            elif isinstance(ts, (list, tuple)) and len(ts) == 4:
                dx = abs(float(ts[1]) - float(ts[0]))
                dy = abs(float(ts[3]) - float(ts[2]))
            else:
                dx = dy = 5.0
            depth = max(dx, dy)
        except Exception:
            depth = 5.0
        self.slot_depth_half = depth * 0.5

        self.slot_forward_heading = math.pi / 2.0  # 위쪽으로 rear-in

        print("[algo] SPECIAL waypoints (top-left):")
        for i, (wx, wy) in enumerate(self.waypoints):
            print(f"  wp[{i}]: ({wx:.2f}, {wy:.2f}) from (x={x:.2f}, y={y:.2f}) yaw={yaw:.2f}")

    # ===================== 경로(웨이포인트) 계산 =====================

    def compute_path(self, obs: Dict[str, Any]) -> None:
        """관측과 맵을 이용해 경로(웨이포인트)를 준비합니다."""

        if self.path_planned:
            return

        # 1) target_slot 중심 계산
        center = self._extract_target_center(obs)
        if center is None:
            print("[algo] no valid target_slot in obs; cannot plan path.")
            return
        self.target_center = center
        cx, cy = center
        print(f"[algo] target_slot center = ({cx:.2f}, {cy:.2f})")

        # 슬롯 인덱스 추정
        self._find_target_slot_index(center)

        # 맵 정보
        if self.map_extent is None:
            return
        xmin, xmax, ymin, ymax = self.map_extent
        height = ymax - ymin
        band_h = height / 3.0

        # 줄 구분
        if cy < ymin + band_h:
            band = "bottom"
        elif cy < ymin + 2 * band_h:
            band = "middle"
        else:
            band = "top"

        # ---- 윗줄 1~3 슬롯 특수 경로 ----
        is_top_band = band == "top"
        if is_top_band and self.target_slot_idx is not None and self.target_slot_idx <= 2:
            print("[algo] using SPECIAL path for top-left slots (idx 0~2)")
            self._compute_special_top_left_path(obs, center)
            return

        # --- 슬롯 폭(한 칸 너비) 계산 ---
        slot_width = 5.0  # 기본값
        ts = obs.get("target_slot")
        try:
            if isinstance(ts, dict):
                slot_width = abs(float(ts["xmax"]) - float(ts["xmin"]))
            elif isinstance(ts, (list, tuple)) and len(ts) == 4:
                slot_width = abs(float(ts[1]) - float(ts[0]))
        except Exception:
            pass

        # 2) 접근 차선 lane_y 계산
        lane_y = self._compute_lane_y(center)
        if lane_y is None:
            print("[algo] cannot compute lane_y; path planning aborted.")
            return
        self.lane_y = lane_y

        # 3) 현재 상태
        state = obs.get("state", {})
        x = float(state.get("x", 0.0))
        y = float(state.get("y", 0.0))
        yaw = float(state.get("yaw", 0.0))

        waypoints: List[Tuple[float, float]] = []

        # (1) y를 lane_y로 맞추기
        if abs(y - lane_y) > 1.0:
            waypoints.append((x, lane_y))

        # ==== pre_turn_x 결정 ====
        if band == "bottom":
            turn_offset = 2.0 * slot_width
        elif band == "middle":
            turn_offset = 1.9 * slot_width
        else:
            turn_offset = 1.9 * slot_width

        SAFE_MARGIN_X = slot_width * 1.2  # 기본

        # 윗줄 왼쪽 1,2 슬롯(인덱스 0,1) — rear-in
        if band == "top" and self.target_slot_idx is not None and self.target_slot_idx <= 1:
            turn_offset = 0.9 * slot_width
            SAFE_MARGIN_X = slot_width * 1.8

        # 아랫줄 왼쪽 1,2,3 슬롯(인덱스 0~2) — front-in
        if band == "bottom" and self.target_slot_idx is not None and self.target_slot_idx <= 2:
            turn_offset = 1.5 * slot_width
            SAFE_MARGIN_X = slot_width * 3.0

        pre_turn_x = cx - turn_offset

        # 왼쪽 벽과 최소 거리 보장
        safe_x = xmin + SAFE_MARGIN_X
        if pre_turn_x < safe_x:
            pre_turn_x = safe_x

        # 현재 위치에서 최소 2m 이상 직진 후 턴
        if pre_turn_x < x + 2.0:
            pre_turn_x = x + 2.0

        print(
            f"[algo] band={band}, slot_idx={self.target_slot_idx}, "
            f"pre_turn_x={pre_turn_x:.2f}, SAFE_X={safe_x:.2f}"
        )

        # (2) 직선 주행 후 턴 시작점
        waypoints.append((pre_turn_x, lane_y))

        # (3) 슬롯 앞쪽 접근 y
        approach_offset = 3.5
        if lane_y < cy:
            approach_y = cy - approach_offset
        else:
            approach_y = cy + approach_offset
        waypoints.append((cx, approach_y))

        # (4) 최종 정지 위치
        stop_ratio = 0.55
        stop_y = approach_y + stop_ratio * (cy - approach_y)
        waypoints.append((cx, stop_y))

        self.waypoints = waypoints
        self.straight_in_mode = False
        self.path_planned = True

        # 슬롯 깊이 계산
        depth = 5.0
        try:
            if isinstance(ts, dict):
                dx = abs(float(ts["xmax"]) - float(ts["xmin"]))
                dy = abs(float(ts["ymax"]) - float(ts["ymin"]))
            elif isinstance(ts, (list, tuple)) and len(ts) == 4:
                dx = abs(float(ts[1]) - float(ts[0]))
                dy = abs(float(ts[3]) - float(ts[2]))
            else:
                dx = dy = 5.0
            depth = max(dx, dy)
        except Exception:
            depth = 5.0
        self.slot_depth_half = depth * 0.5

        # 진입 방향
        if self.lane_y is not None:
            if self.lane_y < cy:
                self.slot_forward_heading = math.pi / 2.0   # 위로
            else:
                self.slot_forward_heading = -math.pi / 2.0  # 아래로
        else:
            self.slot_forward_heading = yaw

        print("[algo] waypoints planned:")
        for i, (wx, wy) in enumerate(self.waypoints):
            print(
                f"  wp[{i}]: ({wx:.2f}, {wy:.2f}) "
                f"from (x={x:.2f}, y={y:.2f}) yaw={yaw:.2f}"
            )

    # ===================== 경로 추종 제어 =====================

    def compute_control(self, obs: Dict[str, Any]) -> Dict[str, float]:
        """경로를 따라가기 위한 조향/가감속 명령을 산출합니다."""
        cmd = {"steer": 0.0, "accel": 0.0, "brake": 0.0, "gear": "D"}

        # 경로가 아직 없으면 한 번 계산
        if not self.path_planned or not self.waypoints:
            self.compute_path(obs)

        state = obs.get("state", {})
        x = float(state.get("x", 0.0))
        y = float(state.get("y", 0.0))
        yaw = float(state.get("yaw", 0.0))  # 라디안
        v = float(state.get("v", 0.0))

        # 웨이포인트가 없다면 그냥 멈춤
        if not self.waypoints:
            cmd["brake"] = 0.8
            return cmd

        # ==== 제어 파라미터 ====
        ANGLE_THRESH_ENTRY      = math.radians(10.0)
        ANGLE_THRESH_STOP  = math.radians(3.0)    # 완전 정지 허용 각도 (더 빡셈)
        DIST_THRESH_STRAIGHT = 2.5

        K_STEER_FAR   = 1.2
        K_STEER_NEAR  = 2.0
        K_STEER_STRAIGHT = 1.5

        SPEED_FAR     = 4.0
        SPEED_MID     = 1.5
        SPEED_NEAR    = 0.7
        SPEED_NEAR_VC = 0.6

        K_SPEED         = 0.6
        WP_REACH_THRESH = 0.4

        # 현재 타깃 웨이포인트
        target_x, target_y = self.waypoints[0]
        dx = target_x - x
        dy = target_y - y
        dist = math.hypot(dx, dy)

        # 웨이포인트 도달 체크
        if dist < WP_REACH_THRESH:
            self.waypoints.pop(0)
            if not self.waypoints:
                self.straight_in_mode = False
                cmd["brake"] = 0.8
                return cmd
            target_x, target_y = self.waypoints[0]
            dx = target_x - x
            dy = target_y - y
            dist = math.hypot(dx, dy)

        # ---- 조향 ----
        target_heading = math.atan2(dy, dx)
        heading_error = _normalize_angle(target_heading - yaw)
        last_segment = len(self.waypoints) == 1
        slot_heading_error = _normalize_angle(self.slot_forward_heading - yaw)

        # straight-in 모드 진입 판단
        if last_segment and not self.straight_in_mode:
            if (dist < DIST_THRESH_STRAIGHT) and (abs(slot_heading_error) < ANGLE_THRESH_ENTRY):
                self.straight_in_mode = True
                print(
                    "[algo] enter straight-in mode "
                    "(dist=%.2f, slot_hdg_err=%.3f)" % (dist, slot_heading_error)
                )
        # 조향 값
        if self.straight_in_mode:
            # 슬롯 방향과의 각도오차를 줄이는 방향으로 약하게 조향
            steer_cmd = K_STEER_STRAIGHT * slot_heading_error
            steer_cmd = max(-1.0, min(1.0, steer_cmd))
        else:
            k_steer = K_STEER_NEAR if last_segment else K_STEER_FAR
            steer_cmd = k_steer * heading_error
            steer_cmd = max(-1.0, min(1.0, steer_cmd))
        cmd["steer"] = steer_cmd

        # ---- 기본 desired speed ----
        if len(self.waypoints) >= 3:
            desired_speed = SPEED_FAR
        elif len(self.waypoints) == 2:
            desired_speed = SPEED_MID
        else:
            desired_speed = SPEED_NEAR

        if len(self.waypoints) == 1 and dist < 1.0:
            desired_speed = SPEED_NEAR_VC

        # ===== straight-in 모드일 때 전방 거리 기반 감속/정지 =====
        if self.straight_in_mode:
            cx = cy = None
            if self.target_center is not None:
                cx, cy = self.target_center

            fx = math.cos(self.slot_forward_heading)
            fy = math.sin(self.slot_forward_heading)

            if cx is not None and cy is not None:
                px = cx - x
                py = cy - y
                dist_forward = px * fx + py * fy
            else:
                dist_forward = 0.0

            STOP_MARGIN_FRONT = 0.10
            STOP_MARGIN_REAR  = 0.25
            SLOW_MARGIN       = 2.0
            MIN_SPEED         = 0.35

            if (self.lane_y is not None) and (cy is not None) and (self.lane_y > cy):
                STOP_MARGIN = STOP_MARGIN_FRONT  # 위→아래 front-in
            else:
                STOP_MARGIN = STOP_MARGIN_REAR   # 아래→위 rear-in

            # 각도도 어느 정도 맞아야 완전히 정지
            well_aligned = abs(slot_heading_error) < ANGLE_THRESH_STOP

            if dist_forward <= 0.0:
                desired_speed = 0.0
            elif dist_forward < STOP_MARGIN and well_aligned:
                # 중심에 거의 왔고 각도도 맞으면 정지
                desired_speed = 0.0
            elif dist_forward < SLOW_MARGIN:
                # 중심에 다가오는 구간에서는 느리게 움직이면서 각도 보정
                desired_speed = max(MIN_SPEED, dist_forward * 0.4)
            else:
                desired_speed = min(desired_speed, SPEED_NEAR)

            # 디버깅 필요하면
            # print(f"[algo] straight: fwd={dist_forward:.2f}, v={v:.2f}, vdes={desired_speed:.2f}")

        # ---- 가감속 명령 ----
        speed_error = desired_speed - v

        if speed_error > 0.05:
            cmd["accel"] = max(0.0, min(1.0, K_SPEED * speed_error))
            cmd["brake"] = 0.0
        elif speed_error < -0.05:
            cmd["accel"] = 0.0
            cmd["brake"] = max(0.0, min(1.0, K_SPEED * (-speed_error)))
        else:
            cmd["accel"] = 0.0
            cmd["brake"] = 0.0

        return cmd


# 전역 planner 인스턴스
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
