"""SOP-Guard 核心引擎：手部關鍵點追蹤（MediaPipe HandLandmarker）、場景條件、SOP 狀態機、開發測試入口。

v2：狀態機每格收一個 FrameContext（手部判定點、物件錨點框、手勢機率），步驟條件分三種：
  hand_dwell / object_in_zone / gesture（定義見 project_store.py）。舊的固定 roi 專案（schema 1）照常可用。

開發測試入口：
    python core_detector.py --project <專案資料夾或名稱> --source <相機編號|影片檔> [--show] [--max-frames N]
                            [--auto-reset-sec 2.0] [--model models/hand_landmarker.task] [--yolo <weights.pt>] [--write-logs]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np

from project_store import Project, Step, app_base_dir, load_project
from scene import Rect, SceneTracker, point_in_rect, rect_overlap_ratio, resolve_zone
from video_recorder import CameraSource, VideoFileSource, is_camera_source, read_image, write_image

HAND_POINT_IDS = (4, 8, 12, 9)   # 拇指尖、食指尖、中指尖、手掌中心（中指根部）
DETECT_LONG_SIDE = 640           # 送進 HandLandmarker 前，影格長邊縮到這個像素
DETECT_EVERY_N_FRAMES = 2        # 相機來源每 N 格做一次手部偵測，其餘格沿用上次結果（處理夠快時）
MONITOR_TORCH_THREADS = 1        # 監控時 PyTorch（YOLO-World）只用 1 個執行緒：手部、物品模型（OpenVINO）、YOLO-World 同時跑會互搶 CPU，
                                 # i5-1235U 實測 torch 1/2/4/6 執行緒 → 10.5～11.3／9.9～10.3／9.8／9.0 fps（離線訓練、建資料集不受影響）


def limit_torch_threads(n: int = MONITOR_TORCH_THREADS) -> None:
    try:
        import torch
        torch.set_num_threads(max(1, int(n)))
    except Exception:
        pass


HAND_MIN_INTERVAL_SEC = 0.06     # 手部偵測的最短間隔（來源時間）：處理快於 16 fps 才隔格做；處理慢（7～10 fps、相機丟格）就每格都做，
                                 # 否則手的取樣會稀到 0.2 秒以上，手停在鞋上這種 0.7 秒的判定會漏
HAND_DET_CONF = 0.4              # MediaPipe 手部偵測／追蹤信心門檻（0.6 時拿東西、手在畫面邊緣常丟手；0.4 實測有手比例 +5～12 點）
DWELL_GRACE_SEC = 0.2            # 條件短暫不成立不超過此秒數，停留計時不歸零
LEAVE_CONFIRM_SEC = 0.3          # hand_and_visual：手離開 ROI 連續達此秒數才做視覺比對
BEEP_FREQ_HZ = 1000
BEEP_DURATION_MS = 400

EVENT_TYPES = ("step_done", "cycle_done", "cycle_discarded", "alert_skip", "alert_timeout", "alert_missing", "reset", "armed")
CSV_HEADER = ["time_iso", "event_type", "step_index", "step_name", "message", "screenshot"]

Point = tuple[float, float]


@dataclass
class FrameContext:
    """狀態機每格需要的觀測。"""
    hand_points: list[Point] = field(default_factory=list)
    hands_xy: np.ndarray | None = None                 # (2, 21, 2) 完整關鍵點，手勢用；可為 None
    anchors: dict[str, list[Rect]] = field(default_factory=dict)
    gesture_probs: dict[str, float] = field(default_factory=dict)
    box_state: str | None = None                       # 盒內狀態辨識的目前狀態（StateStream.current）
    items_present: set = field(default_factory=set)    # 物品偵測：這一格盒內看得到哪些物品（物件版）
    box_open: bool | None = None                       # 盒子開著(True)／蓋著(False)／不知道(None)；蓋著視同盒子不在
    box_seen: bool = False                             # 固定桌面範圍：訓練的模型看到「盒子的樣子」（立著／翻面／鞋子放在上面）→ 盒子還在


# ---------------------------------------------------------------- 幾何與影像

def any_point_in_rect(points, rect) -> bool:
    return any(point_in_rect(p, rect) for p in points)


def any_point_in_rects(points, rects) -> bool:
    return any(any_point_in_rect(points, r) for r in rects)


def crop_roi(frame, roi):
    x, y, w, h = [int(round(v)) for v in roi]
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    return frame[y0:y1, x0:x1]


def roi_similarity(img_a, img_b) -> float:
    """HSV 的 H×S 二維直方圖相關係數，夾到 0～1。"""
    if img_a is None or img_b is None or img_a.size == 0 or img_b.size == 0:
        return 0.0
    h, w = img_a.shape[:2]
    b = cv2.resize(img_b, (w, h), interpolation=cv2.INTER_AREA)
    hsv_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
    hist_a = cv2.calcHist([hsv_a], [0, 1], None, [30, 32], [0, 180, 0, 256])
    hist_b = cv2.calcHist([hsv_b], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist_a, hist_a, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_b, hist_b, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    v = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
    if v != v:
        return 0.0
    return float(min(max(v, 0.0), 1.0))


def play_alert_sound() -> None:
    """winsound.Beep 會阻塞，放到 daemon thread；非 Windows 環境只印訊息。"""
    def _beep():
        try:
            import winsound
            winsound.Beep(BEEP_FREQ_HZ, BEEP_DURATION_MS)
        except Exception:
            print("[警告音] 此環境沒有 winsound，略過")
    threading.Thread(target=_beep, daemon=True).start()


def union_rect(rects: list[Rect]) -> Rect | None:
    if not rects:
        return None
    x1 = min(r[0] for r in rects); y1 = min(r[1] for r in rects)
    x2 = max(r[0] + r[2] for r in rects); y2 = max(r[1] + r[3] for r in rects)
    return (x1, y1, x2 - x1, y2 - y1)


def state_region(project: Project, anchors: dict[str, list[Rect]]) -> Rect | None:
    """盒內狀態要裁的區域：有鎖定的固定區域就用它；否則用 state_anchor 的框套 state_rel。沒有就 None。
    錨點有多個框時（打開的盒子常同時偵測到「整個盒子含盒蓋」和「盒身」）取最內層那個：包住別的候選框的不要，
    內容物在盒身裡，裁到盒蓋只會多一塊紙板。剩下多個就照錨點 pick 的順序取第一個。"""
    from scene import rect_overlap_ratio, sub_rect
    if project.state_fixed:
        return tuple(project.state_fixed)
    if project.state_anchor:
        rects = anchors.get(project.state_anchor, [])
        if not rects:
            return None
        inner = [r for r in rects if not any(o is not r and rect_overlap_ratio(o, r) >= 0.9 for o in rects)]
        return sub_rect((inner or rects)[0], project.state_rel)
    return None


CONTINUOUS_GAP_SEC = 0.3         # continuous 步驟：條件中斷不超過這麼久仍算同一段連續停留（偵測每 5～15 格一次會閃）
STATE_INNER_SHRINK = 0.15        # 手「在盒內」的判定只看區域往內縮 15% 的範圍：手扶著盒緣、蓋蓋子不算遮住內容
STATE_FRESH_SEC = 1.0            # 錨點超過這麼久沒再真的偵測到就不判（沿用中的舊框可能已對著空桌面）
STATE_SETTLE_SEC = 0.5           # 盒子位置剛變（被推走／剛放上來）後等這麼久再判：移動中、半出畫面的盒子會被判成別的狀態


def hands_in_inner(hand_points, region: Rect, shrink: float = STATE_INNER_SHRINK, need: int = 2) -> bool:
    """某一隻手有 ≥ need 個關鍵點落在區域內縮後的範圍 → 手伸進盒內（會遮住內容物）。
    hand_points 每手 4 點（HAND_POINT_IDS 的順序），依序分組。"""
    x, y, w, h = region
    inner = (x + w * shrink, y + h * shrink, w * (1 - 2 * shrink), h * (1 - 2 * shrink))
    n = len(HAND_POINT_IDS)
    for i in range(0, len(hand_points), n):
        if sum(point_in_rect(q, inner) for q in hand_points[i:i + n]) >= need:
            return True
    return False


class BoxStateRunner:
    """把「裁區域 → 判狀態（每 N 格）→ 手伸進盒內不判 → 盒子不見就清掉狀態」包起來，給監控迴圈與離線工具共用。"""

    def __init__(self, project: Project):
        from state_recognizer import StateLibrary, StateResult, StateStream
        self.project = project
        self.lib = StateLibrary(project.dir_path)
        self.stream = StateStream(self.lib)
        self._result = StateResult
        self.every = max(1, project.state_every_n_frames)
        self._n = 0
        self._last_run_t: float | None = None     # 上次真的跑偵測的來源時間（有給 t_now 時用）
        self._moved_t: float | None = None        # 區域上次換位置的來源時間
        self._moved_at = -10**9                   # 區域上次換位置是第幾格
        self._settle = max(1, int(round(STATE_SETTLE_SEC * (project.fps or 30.0))))
        self.region: Rect | None = None
        self.last_age: float | None = None        # 區域對應的錨點距上次真的偵測到幾秒（畫面上區分「現在看到」與「沿用」）
        self.debug: dict | None = None            # 這一格有做判定時的紀錄（--state-log 用）
        # 物件版：專案有訓練好的物品模型就一起跑（同一張盒內裁圖）
        self.items = None
        self.items_present: dict[str, float] = {}
        self.item_hits: list = []
        from item_recognizer import ItemDetector, model_path
        mp = model_path(project)
        # 只在專案真的用物品步驟時才載入（手勢模式與物品模式二選一；留著舊模型檔不該讓它白跑）
        self.open_closed = None
        self.box_open: bool | None = None
        self._open_votes: list[bool] = []
        # 固定桌面範圍（state_fixed）：不管盒子在不在、每 N 格就在固定裁圖上偵測；看到 box_labels 裡的物品就當盒子還在
        from item_recognizer import ItemSeeds, OpenClosedClassifier, is_fixed
        seeds = ItemSeeds(project.dir_path)
        self.fixed_items = is_fixed(project)      # 有物品模型時才真的用到（沒有模型就走原本的盒內狀態路徑）
        self.box_labels: set[str] = set(seeds.box_labels)
        self.box_seen = False
        if os.path.isfile(mp) and any(s.type == "item" for s in project.steps):
            self.items = ItemDetector(mp)
            if not self.fixed_items:
                oc = OpenClosedClassifier(project, seeds)
                if oc.enabled:
                    self.open_closed = oc
        # 狀態辨識只在步驟或結束條件真的用到 box_state 時才跑：物件版專案常留著舊的狀態範例，白跑 CLIP 每 5 格 60 ms 會拖慢相機
        uses_state = any(s.type == "box_state" for s in project.steps) or (project.cycle_end is not None and project.cycle_end.type == "box_state")
        if not uses_state:
            self.lib.examples = []

    @property
    def enabled(self) -> bool:
        return bool(self.lib.examples) or self.items is not None

    def _due(self, t_now: float | None) -> bool:
        """這一格要不要跑偵測：有來源時間就以「每 every 格 ÷ fps 秒」為間隔（處理慢、相機丟格時自動變成每格都跑，
        取樣密度才不會跟著處理速度掉——現場影片網頁實跑只處理 1/3 的格，盒底朝上只露 1 秒就漏掉）；沒有時間就照格數。"""
        if t_now is None:
            return self._n % self.every == 0
        interval = self.every / float(self.project.fps or 30.0)
        if self._last_run_t is None or t_now - self._last_run_t >= interval - 1e-6:
            self._last_run_t = t_now
            return True
        return False

    def update(self, frame_bgr, anchors: dict[str, list[Rect]], hand_points, anchor_age: float | None = None,
               t_now: float | None = None) -> str | None:
        """anchor_age：盒子錨點距上次真的偵測到過了幾秒（SceneTracker.age）；None = 不知道，當作新鮮。
        t_now：來源時間（秒）；給了就用時間決定偵測與停好的間隔，不受處理速度影響。"""
        from scene import rect_overlap_ratio
        self._n += 1
        self.debug = None
        if self.fixed_items and self.items is not None:
            # 看整張桌面：範圍固定、不看錨點（翻鞋舌檢查在盒子放上來之前、鞋子放到盒上後盒子會被遮住）；手在不在也照偵測
            self.region = tuple(self.project.state_fixed); self.last_age = None
            if not self._due(t_now) or frame_bgr is None:
                return self.stream.current
            self.item_hits = self.items.detect(crop_roi(frame_bgr, self.region))
            self.items_present = {h.name: h.conf for h in self.item_hits}
            self.box_seen = any(h.name in self.box_labels for h in self.item_hits)
            self.debug = {"why": "judged", "items": "|".join(sorted(self.items_present)), "box_seen": self.box_seen}
            return self.stream.current
        new = state_region(self.project, anchors)
        if new is None:
            # 盒子不在（錨點過期／被拿走）→ 狀態清成「不知道」。不能維持上一次：上一輪的「蓋上」會被下一輪當結束條件，
            # 沿用中的舊框對著空桌面判出來的「空盒」也會讓下一輪提早開始
            if self.region is not None:
                self.stream.reset()
                self.debug = {"why": "box_gone"}
            self.region = None
            self.items_present = {}; self.item_hits = []
            self.box_open = None; self._open_votes = []
            return self.stream.push(None)
        if self.region is None or min(rect_overlap_ratio(new, self.region), rect_overlap_ratio(self.region, new)) < 0.6:
            # 一框包住另一框（偵測在「整盒含盒蓋」和「盒身」之間切換）是同一個沒動的盒子，不算移動
            nested = self.region is not None and max(rect_overlap_ratio(new, self.region), rect_overlap_ratio(self.region, new)) >= 0.9
            if not nested:
                self._moved_at = self._n; self._moved_t = t_now
        self.region = new                           # 區域永遠跟著最新偵測（之前黏住舊框，盒子被挪 30% 畫面上的框就對不上）
        self.last_age = anchor_age
        if not self._due(t_now):
            return self.stream.current
        if frame_bgr is None:
            return self.stream.push(None)
        settling = (t_now - self._moved_t < STATE_SETTLE_SEC) if (t_now is not None and self._moved_t is not None) else (self._n - self._moved_at < self._settle)
        if settling:
            # 盒子剛換位置（被推走、剛放上來）：移動中／半出畫面的盒子會被判成別的狀態，等它停好再判
            self.stream.last = self._result(self.stream.current, reason="盒子移動中")
            self.debug = {"why": "moving"}
            return self.stream.current
        if anchor_age is not None and anchor_age > STATE_FRESH_SEC:
            self.stream.last = self._result(self.stream.current, reason="盒子暫時沒偵測到")
            self.debug = {"why": "stale", "age": round(anchor_age, 2)}
            return self.stream.current
        hands_inside = hands_in_inner(hand_points, self.region)
        crop = crop_roi(frame_bgr, self.region)
        if self.open_closed is not None and not hands_inside:
            # 開著／蓋著：最近 3 次投票（蓋蓋子途中會閃）
            o = self.open_closed.is_open(crop)
            if o is not None:
                self._open_votes = (self._open_votes + [o])[-3:]
                if len(self._open_votes) == 3 and len(set(self._open_votes)) == 1:
                    self.box_open = self._open_votes[0]
                elif self.box_open is None:
                    self.box_open = o
        if self.items is not None and self.box_open is not False:
            # 物品偵測不等手離開：手一直在盒內、放完就蓋上的話，等手離開就沒機會看到了。被手遮住的那一樣本來就認不到，
            # 看得到的照認；步驟要連續 0.5 秒看到才算，手上拿著晃過去的通常撐不到。蓋著的盒子裡面看不到，不跑
            self.item_hits = self.items.detect(crop)
            self.items_present = {h.name: h.conf for h in self.item_hits}
        elif self.box_open is False:
            self.item_hits = []; self.items_present = {}
        if hands_inside:
            self.debug = {"why": "hands", "items": "|".join(sorted(self.items_present))}
            return self.stream.push(None, hands_inside=True)
        if not self.lib.examples:
            self.debug = {"why": "judged", "items": "|".join(sorted(self.items_present))}
            return self.stream.current
        cur = self.stream.push(crop)
        r = self.stream.last
        self.debug = {"why": "judged", "raw": r.state, "conf": round(r.confidence, 3), "margin": round(r.margin, 3), "items": "|".join(sorted(self.items_present))}
        return cur


# ---------------------------------------------------------------- 手部追蹤

class HandTracker:
    """MediaPipe Tasks API HandLandmarker（VIDEO 模式）。"""

    def __init__(self, model_path: str, num_hands: int = 2,
                 min_detection_confidence: float = HAND_DET_CONF, min_tracking_confidence: float = HAND_DET_CONF):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"找不到手部模型檔：{os.path.abspath(model_path)}（請依 models/README.md 下載）")
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1
        self.last_points: list[Point] = []
        self.last_hands_xy: np.ndarray = np.full((2, 21, 2), np.nan, np.float32)

    def detect(self, frame_bgr, timestamp_ms: int) -> list[Point]:
        fh, fw = frame_bgr.shape[:2]
        scale = DETECT_LONG_SIDE / max(fw, fh)
        if scale < 1.0:
            small = cv2.resize(frame_bgr, (max(1, int(fw * scale)), max(1, int(fh * scale))), interpolation=cv2.INTER_AREA)
        else:
            small = frame_bgr
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        ts = int(timestamp_ms)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        result = self._detector.detect_for_video(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), ts)
        points: list[Point] = []
        hands_xy = np.full((2, 21, 2), np.nan, np.float32)
        for k, hand in enumerate(result.hand_landmarks[:2]):
            hands_xy[k] = [(lm.x * fw, lm.y * fh) for lm in hand]
            for i in HAND_POINT_IDS:
                lm = hand[i]
                points.append((lm.x * fw, lm.y * fh))
        self.last_points = points
        self.last_hands_xy = hands_xy
        return points

    def close(self) -> None:
        self._detector.close()


# ---------------------------------------------------------------- 狀態機

class SOPStateMachine:
    """停留／跳步／超時／重置／武裝的規則不分步驟類型；每格先把每個步驟的條件算成 True/False。"""

    def __init__(self, project: Project, error_log_dir: str | None = None, sound_enabled: bool = True):
        self.project = project
        self.steps: list[Step] = project.steps
        if not self.steps:
            raise ValueError("專案沒有任何步驟")
        self.zone_specs = project.zone_specs()
        self.error_log_dir = error_log_dir
        self.sound_enabled = sound_enabled
        self.events: list[dict] = []
        self.cycle_count = 0
        self.bad_cycle_count = 0                 # 清單模式：結束時有漏做的輪數
        self.mode = project.mode
        self.observed: set[int] = set()          # 清單模式：本輪已達標的步驟
        self.acc_sec: dict[int, float] = {}      # 清單模式：本輪各步驟累計成立秒數
        self.streak_sec: dict[int, float] = {}   # 清單模式 continuous 步驟：目前這一段連續成立了幾秒
        self.gap_sec: dict[int, float] = {}      # 　同上：這一段中斷了幾秒（超過 CONTINUOUS_GAP_SEC 就從頭算）
        self.cycle_active = False
        self.cycle_started_at = 0.0
        self.end_hold_sec = 0.0
        self.end_seen = False                    # 清單模式 object_absent：這一輪有沒有真的看到過那個物件（看到過才有「拿走」可言）
        self.alert_active = False
        self.alert_message = ""
        self.alert_type = ""
        self.alert_started_at = 0.0
        self.last_similarity: float | None = None
        self.armed = True
        self.clear_sec = 0.0
        self.reset_dwell_sec_acc = 0.0
        self.reset_zone_armed = True
        self._t_prev: float | None = None
        self._t_now = 0.0
        self._templates: dict[int, object] = {}
        self._last_zones: dict[int, list[Rect]] = {}
        self._reset_cycle_state(0.0)
        if error_log_dir:
            os.makedirs(error_log_dir, exist_ok=True)

    # ---- 內部狀態

    def _reset_cycle_state(self, t_now: float) -> None:
        self.current_step_idx = 0
        self.phase = "waiting_hand"
        self.dwell_sec = 0.0
        self.miss_sec = 0.0
        self.leave_sec = 0.0
        self.step_started_at = t_now
        self.foreign_dwell: dict[int, float] = {}
        self.foreign_miss: dict[int, float] = {}

    def _template(self, step: Step):
        if step.step_index not in self._templates:
            path = os.path.join(self.project.dir_path, step.template_file) if step.template_file else ""
            self._templates[step.step_index] = read_image(path) if path and os.path.isfile(path) else None
        return self._templates[step.step_index]

    def _zones(self, step: Step, ctx: FrameContext) -> list[Rect]:
        if step.type == "hand_on_object":
            return list(ctx.anchors.get(step.object or "", []))     # 畫面上畫的區域＝那個物件現在的框
        if not step.zone:
            return []
        return resolve_zone(self.zone_specs[step.zone], ctx.anchors)

    def _condition(self, step: Step, ctx: FrameContext) -> bool:
        if step.type == "hand_dwell":
            return any_point_in_rects(ctx.hand_points, self._zones(step, ctx))
        if step.type == "object_in_zone":
            objs = ctx.anchors.get(step.object or "", [])
            zones = self._zones(step, ctx)
            if not objs or not zones:
                return False
            inside = [max(rect_overlap_ratio(o, z) for z in zones) >= step.min_overlap for o in objs]
            return all(inside) if step.count == "all" else any(inside)
        if step.type == "gesture":
            # 這一類要是目前最高分、而且機率 ≥ min_prob（機率會分散在各類，只看絕對值會太嚴）
            p = ctx.gesture_probs.get(step.label or "", 0.0)
            return p >= step.min_prob and p >= max(ctx.gesture_probs.values(), default=0.0)
        if step.type == "object_absent":
            # 蓋著的盒子 YOLO 一樣認成盒子；物件版有標「蓋著」範例時，蓋著就當作盒子已經不在（這一輪結束）。
            # 固定桌面範圍：鞋子放到盒上後 YOLO 常找不到盒子，訓練的模型看到「盒子的樣子」也算在（box_seen）
            return ((not ctx.anchors.get(step.object or "", [])) and not ctx.box_seen) or ctx.box_open is False
        if step.type == "hands_absent":
            return not ctx.hand_points
        if step.type == "box_state":
            return ctx.box_state is not None and ctx.box_state == step.label
        if step.type == "item":
            return (step.label or "") in ctx.items_present
        if step.type == "hand_on_object":
            # 手停在某個物件（通用偵測的鞋子）上：某隻手 4 個關鍵點裡 ≥ 2 個落在物件框內。框跟著物件走，不綁固定座標，
            # 換鞋款、鏡頭偏一點都不用改；手握住鞋子時偵測會掉框，靠錨點 max_age_sec 沿用（現場影片：沿用 2.5 秒後
            # 檢查片段 97% 的取樣手在鞋上、靜置時 5%）
            return any(hands_in_inner(ctx.hand_points, r, shrink=0.0, need=2) for r in ctx.anchors.get(step.object or "", []))
        return False

    def _log(self, event_type: str, step: Step | None, message: str, screenshot: str = "") -> None:
        ev = {
            "t": self._t_now,
            "time_iso": datetime.now().isoformat(timespec="seconds"),
            "event_type": event_type,
            "step_index": "" if step is None else step.step_index,
            "step_name": "" if step is None else step.step_name,
            "message": message,
            "screenshot": screenshot,
        }
        self.events.append(ev)
        if self.error_log_dir is not None:
            path = os.path.join(self.error_log_dir, "events.csv")
            is_new = not os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8-sig" if is_new else "utf-8") as f:
                writer = csv.writer(f)
                if is_new:
                    writer.writerow(CSV_HEADER)
                writer.writerow([ev[k] for k in CSV_HEADER])

    def _check_reset_zone(self, hand_points, dt: float) -> None:
        zone = self.project.reset_zone
        if not zone:
            return
        if any_point_in_rect(hand_points, zone):
            self.reset_dwell_sec_acc += dt
            if self.reset_dwell_sec_acc >= self.project.reset_dwell_sec and self.reset_zone_armed:
                self.reset()
                self.reset_zone_armed = False
        else:
            self.reset_dwell_sec_acc = 0.0
            self.reset_zone_armed = True

    def _trigger_alert(self, event_type: str, message: str, frame_bgr) -> None:
        self.alert_active = True
        self.phase = "alert"
        self.alert_type = event_type
        self.alert_message = message
        self.alert_started_at = self._t_now
        if self.sound_enabled:
            play_alert_sound()
        screenshot = ""
        if self.error_log_dir is not None and frame_bgr is not None:
            kind = {"alert_skip": "skip", "alert_timeout": "timeout"}.get(event_type, "missing")
            screenshot = f"{datetime.now():%Y%m%d_%H%M%S}_{kind}_{self.current_step_idx}.jpg"
            write_image(os.path.join(self.error_log_dir, screenshot), frame_bgr)
        self._log(event_type, None if self.mode == "checklist" else self.steps[self.current_step_idx], message, screenshot)

    def _complete_step(self, t_now: float) -> None:
        cur = self.steps[self.current_step_idx]
        self._log("step_done", cur, cur.step_name)
        self.current_step_idx += 1
        self.dwell_sec = self.miss_sec = self.leave_sec = 0.0
        self.phase = "waiting_hand"
        self.step_started_at = t_now
        self.foreign_dwell = {}
        self.foreign_miss = {}
        if self.current_step_idx >= len(self.steps):
            self.cycle_count += 1
            self._log("cycle_done", None, f"#{self.cycle_count}")
            self.current_step_idx = 0
            self.armed = False
            self.clear_sec = 0.0

    # ---- 公開介面

    def update(self, frame_bgr, ctx, t_now: float) -> None:
        if not isinstance(ctx, FrameContext):
            ctx = FrameContext(hand_points=list(ctx))
        dt = 0.0 if self._t_prev is None else max(0.0, t_now - self._t_prev)
        self._t_prev = t_now
        self._t_now = t_now
        conds = [self._condition(s, ctx) for s in self.steps]
        self._last_zones = {s.step_index: self._zones(s, ctx) for s in self.steps}
        if self.mode == "checklist":
            self._update_checklist(frame_bgr, ctx, conds, dt, t_now)
        else:
            self._update_ordered(frame_bgr, ctx, conds, dt, t_now)

    def _update_checklist(self, frame_bgr, ctx: FrameContext, conds: list[bool], dt: float, t_now: float) -> None:
        """清單模式：步驟不分先後、各自累計成立時間；一輪由 cycle_end 條件結束，結束時缺的步驟一次報出。"""
        end = self.project.cycle_end
        end_now = self._condition(end, ctx)
        gated = end.type == "object_absent" and self.project.state_fixed is not None
        if gated:
            # 看整張桌面時「盒子被拿走」要先在這一輪看到過盒子才算：翻鞋舌檢查發生在盒子放上來之前，那時盒子本來就不在，
            # 不能當成結束、也不能擋住下一輪開始；看到過（end_seen）之後不在才是拿走。只在輪次進行中計，輪與輪之間
            # 偶然看到盒子（空檔時 YOLO 閃一下、別處的藍色東西）不算——現場影片第 6 輪就是這樣被擋住：空檔 84 秒、157 秒
            # 各閃了一次，接下來的翻鞋舌檢查被當成「結束條件成立中」沒開始新的一輪，整輪報漏做。
            # 跟著盒子裁的模式維持原樣（蓋著的盒子放著不動時不武裝，下一輪要等它被拿走）
            if not self.cycle_active:
                end_now = False
            else:
                if not end_now:
                    self.end_seen = True
                end_now = end_now and self.end_seen
        if self.alert_active:
            self._check_reset_zone(ctx.hand_points, dt)
            return
        self._check_reset_zone(ctx.hand_points, dt)
        if not self.armed:
            # 等結束條件解除（例如鞋子回到工作區）一段時間，下一輪才開始
            if end_now:
                self.clear_sec = 0.0
            else:
                self.clear_sec += dt
                if self.clear_sec >= self.project.cycle_arm_clear_sec:
                    self.armed = True
                    self._log("armed", None, "")
            return
        for s, ok in zip(self.steps, conds):
            i = s.step_index
            if s.continuous:
                # 連續停留：中斷 ≤ CONTINUOUS_GAP_SEC 不算斷（偵測每幾格一次會閃）；斷久了這一段歸零。
                # 手經過鞋子的零碎片段（靜置時約 5% 取樣）整輪累加也會超過門檻，所以不能用累加
                if ok:
                    self.streak_sec[i] = self.streak_sec.get(i, 0.0) + dt; self.gap_sec[i] = 0.0
                else:
                    self.gap_sec[i] = self.gap_sec.get(i, 0.0) + dt
                    if self.gap_sec[i] > CONTINUOUS_GAP_SEC:
                        self.streak_sec[i] = 0.0
                ok = self.streak_sec.get(i, 0.0) >= s.dwell_time_sec
                if i not in self.observed:
                    self.acc_sec[i] = self.streak_sec.get(i, 0.0)       # 畫面進度條用
            if not ok:
                continue
            if not self.cycle_active:
                if end_now:
                    continue   # 結束條件成立中（例如鞋子不在工作區）不算開始新的一輪
                self.cycle_active = True
                self.cycle_started_at = t_now
                if gated:
                    self.end_seen = bool(ctx.anchors.get(end.object or "", [])) or ctx.box_seen
            if not s.continuous:
                self.acc_sec[i] = self.acc_sec.get(i, 0.0) + dt
            if i not in self.observed and self.acc_sec.get(i, 0.0) >= s.dwell_time_sec:
                self.observed.add(i)
                self._log("step_done", s, s.step_name)
        self.end_hold_sec = self.end_hold_sec + dt if end_now else 0.0
        if not self.cycle_active:
            return
        if self.end_hold_sec >= end.dwell_time_sec:
            missing = [s for s in self.steps if s.required and s.step_index not in self.observed]
            cycle_len = t_now - self.cycle_started_at
            if cycle_len < self.project.cycle_min_sec:
                # 太短不算一輪（例如只是把鞋拿出畫面）：不計數、不報，直接重新等下一輪
                self._log("cycle_discarded", None, f"只持續 {cycle_len:.1f} 秒")
                self.observed = set(); self.acc_sec = {}; self.streak_sec = {}; self.gap_sec = {}; self.cycle_active = False; self.end_hold_sec = 0.0
                self.armed = False; self.clear_sec = 0.0; self.end_seen = False
                return
            if missing:
                self.bad_cycle_count += 1
                self._trigger_alert("alert_missing", "漏做：" + "、".join(s.step_name for s in missing), frame_bgr)
            else:
                self.cycle_count += 1
                self._log("cycle_done", None, f"#{self.cycle_count}")
            self.observed = set(); self.acc_sec = {}; self.streak_sec = {}; self.gap_sec = {}; self.cycle_active = False; self.end_hold_sec = 0.0
            self.armed = False; self.clear_sec = 0.0; self.end_seen = False
            return
        if self.project.cycle_timeout_sec and (t_now - self.cycle_started_at) > self.project.cycle_timeout_sec:
            missing = [s for s in self.steps if s.required and s.step_index not in self.observed]
            self._trigger_alert("alert_timeout", "本輪超時，尚未完成：" + ("、".join(s.step_name for s in missing) or "（無）"), frame_bgr)

    def _update_ordered(self, frame_bgr, ctx: FrameContext, conds: list[bool], dt: float, t_now: float) -> None:
        # 1. 警告中：只看重置區
        if self.alert_active:
            self._check_reset_zone(ctx.hand_points, dt)
            return

        # 2. 重置區
        self._check_reset_zone(ctx.hand_points, dt)

        # 2b. 等所有步驟條件都不成立
        if not self.armed:
            if any(conds):
                self.clear_sec = 0.0
            else:
                self.clear_sec += dt
                if self.clear_sec >= self.project.cycle_arm_clear_sec:
                    self.armed = True
                    self._log("armed", None, "")
            return

        # 3. 跳步檢查
        cur_idx = self.current_step_idx
        for j in range(cur_idx + 1, len(self.steps)):
            if conds[j]:
                self.foreign_dwell[j] = self.foreign_dwell.get(j, 0.0) + dt
                self.foreign_miss[j] = 0.0
            else:
                self.foreign_miss[j] = self.foreign_miss.get(j, 0.0) + dt
                if self.foreign_miss[j] >= DWELL_GRACE_SEC:
                    self.foreign_dwell[j] = 0.0
            if self.foreign_dwell.get(j, 0.0) >= self.project.skip_alert_sec:
                names = "、".join(s.step_name for s in self.steps[cur_idx:j])
                self._trigger_alert("alert_skip", f"遺漏步驟：{names}", frame_bgr)
                return

        # 4. 超時檢查（步驟 0 不算）
        cur = self.steps[cur_idx]
        if cur_idx > 0 and (t_now - self.step_started_at) > cur.timeout_sec:
            self._trigger_alert("alert_timeout", f"步驟超時：{cur.step_name}", frame_bgr)
            return

        # 5. 當前步驟推進
        active = conds[cur_idx]
        if self.phase == "waiting_hand":
            if active:
                self.dwell_sec += dt
                self.miss_sec = 0.0
            else:
                self.miss_sec += dt
                if self.miss_sec >= DWELL_GRACE_SEC:
                    self.dwell_sec = 0.0
            if self.dwell_sec >= cur.dwell_time_sec:
                if cur.trigger_mode == "hand_only":
                    self._complete_step(t_now)
                else:
                    self.phase = "waiting_leave"
                    self.leave_sec = 0.0
        elif self.phase == "waiting_leave":
            if active:
                self.leave_sec = 0.0
            else:
                self.leave_sec += dt
                if self.leave_sec >= LEAVE_CONFIRM_SEC:
                    zones = self._last_zones.get(cur_idx) or []
                    sim = roi_similarity(crop_roi(frame_bgr, zones[0]), self._template(cur)) if zones and frame_bgr is not None else 0.0
                    self.last_similarity = sim
                    if sim >= (cur.visual_match_min if cur.visual_match_min is not None else 0.0):
                        self._complete_step(t_now)

    def reset(self) -> None:
        self._reset_cycle_state(self._t_now)
        self.observed = set(); self.acc_sec = {}; self.streak_sec = {}; self.gap_sec = {}; self.cycle_active = False; self.end_hold_sec = 0.0; self.end_seen = False
        self.alert_active = False
        self.alert_message = ""
        self.alert_type = ""
        self.armed = False
        self.clear_sec = 0.0
        self.reset_dwell_sec_acc = 0.0
        self._log("reset", None, "")

    def snapshot(self) -> dict:
        cur = self.steps[self.current_step_idx]
        states = []
        for i in range(len(self.steps)):
            if self.mode == "checklist":
                states.append("done" if i in self.observed else "pending")
            else:
                states.append("done" if i < self.current_step_idx else "active" if i == self.current_step_idx else "pending")
        return {
            "mode": self.mode,
            "bad_cycle_count": self.bad_cycle_count,
            "step_states": states,
            "current_step_idx": self.current_step_idx,
            "progress": min(self.dwell_sec / cur.dwell_time_sec, 1.0) if cur.dwell_time_sec > 0 else 0.0,
            "phase": self.phase,
            "armed": self.armed,
            "alert_active": self.alert_active,
            "alert_message": self.alert_message,
            "cycle_count": self.cycle_count,
            "last_similarity": self.last_similarity,
            "zones": {i: list(z) for i, z in self._last_zones.items()},
        }


# ---------------------------------------------------------------- 開發預覽視窗

_CJK_FONT_PATHS = [r"C:\Windows\Fonts\msjh.ttc", r"C:\Windows\Fonts\msjhbd.ttc", r"C:\Windows\Fonts\mingliu.ttc"]
_cjk_fonts: dict[int, object] = {}


def _cjk_font(size: int):
    """開發預覽用：用 PIL 畫中文（正式 UI 走 QPainter）。找不到字型就回 None，改用 ASCII。"""
    if size in _cjk_fonts:
        return _cjk_fonts[size]
    font = None
    try:
        from PIL import ImageFont
        for p in _CJK_FONT_PATHS:
            if os.path.isfile(p):
                font = ImageFont.truetype(p, size)
                break
    except Exception:
        font = None
    _cjk_fonts[size] = font
    return font


_text_cache: dict[tuple, tuple] = {}      # (text, size, color) → (bgr 小圖, alpha 0～1)；同一行字只用 PIL 畫一次
_TEXT_CACHE_MAX = 3000


def _render_text(text: str, size: int, color) -> tuple:
    from PIL import Image, ImageDraw
    key = (text, size, tuple(color))
    hit = _text_cache.get(key)
    if hit is not None:
        return hit
    if len(_text_cache) >= _TEXT_CACHE_MAX:
        _text_cache.clear()
    font = _cjk_font(size); l, t, r_, b = font.getbbox(text)
    w, h = max(1, r_ + 2), max(1, b + 2)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0)); draw = ImageDraw.Draw(img)
    r, g, b_ = color[2], color[1], color[0]
    draw.text((1, 1), text, font=font, fill=(0, 0, 0, 255)); draw.text((0, 0), text, font=font, fill=(r, g, b_, 255))
    arr = np.asarray(img)
    hit = (np.ascontiguousarray(arr[..., 2::-1]), arr[..., 3:4].astype(np.float32) / 255.0)
    _text_cache[key] = hit
    return hit


def draw_texts(frame, items: list[tuple[int, int, str, tuple[int, int, int], int]]):
    """items: (x, y, text, color_bgr, size)。每一行字第一次用 PIL 畫成小圖快取，之後只做 numpy 貼圖：
    疊字面板十幾行每格從約 19 ms 降到 2～3 ms（整張畫面 PIL 來回轉換才是原本最貴的部分）。"""
    if not items:
        return frame
    if _cjk_font(items[0][4]) is None:
        for x, y, text, color, size in items:
            cv2.putText(frame, text.encode("ascii", "replace").decode(), (x, y + size), cv2.FONT_HERSHEY_SIMPLEX, size / 28, color, 2)
        return frame
    fh, fw = frame.shape[:2]
    for x, y, text, color, size in items:
        if not text:
            continue
        bgr, a = _render_text(text, size, color)
        h, w = bgr.shape[:2]
        x1, y1 = max(0, x), max(0, y); x2, y2 = min(fw, x + w), min(fh, y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        sx, sy = x1 - x, y1 - y
        roi = frame[y1:y2, x1:x2]; src = bgr[sy:sy + (y2 - y1), sx:sx + (x2 - x1)]; al = a[sy:sy + (y2 - y1), sx:sx + (x2 - x1)]
        roi[:] = (roi * (1.0 - al) + src * al).astype(np.uint8)
    return frame


def _draw_dev_overlay(frame, project: Project, fsm: SOPStateMachine, ctx: FrameContext, fps_text: str, state_runner=None):
    snap = fsm.snapshot()
    scale = 960 / max(frame.shape[:2])
    if scale < 1.0:
        frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
    else:
        scale = 1.0
    S = lambda v: int(v * scale)  # noqa: E731
    colors = {"pending": (0, 200, 255), "active": (255, 120, 0), "done": (0, 200, 0)}  # BGR
    texts: list[tuple[int, int, str, tuple[int, int, int], int]] = []
    for name, rects in ctx.anchors.items():
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (S(x), S(y)), (S(x + w), S(y + h)), (200, 200, 200), 2)
            texts.append((S(x) + 4, S(y) + 2, name, (230, 230, 230), 18))
    for s, state in zip(project.steps, snap["step_states"]):
        for (x, y, w, h) in snap["zones"].get(s.step_index, []):
            cv2.rectangle(frame, (S(x), S(y)), (S(x + w), S(y + h)), colors[state], 3)
            texts.append((S(x) + 4, S(y + h) - 26, f"{s.step_index + 1}.{s.step_name}", colors[state], 20))
            if state == "active":
                cv2.rectangle(frame, (S(x), S(y + h) + 4), (S(x) + int(S(w) * snap["progress"]), S(y + h) + 12), colors[state], -1)
    if project.reset_zone:
        x, y, w, h = project.reset_zone
        cv2.rectangle(frame, (S(x), S(y)), (S(x + w), S(y + h)), (255, 0, 180), 3)
        texts.append((S(x) + 4, S(y) + 2, "重置區", (255, 0, 180), 18))
    for px, py in ctx.hand_points:
        cv2.circle(frame, (S(px), S(py)), 5, (0, 255, 0), -1)

    # 狀態面板：畫在畫面「下方」（使用者裁示；原本在左上角會蓋住紙箱那一帶）。先把每一行收集起來算高度，再畫底色
    fh, fw = frame.shape[:2]
    panel: list[tuple[int, int, str, tuple[int, int, int], int]] = []       # (x, 相對 y, 文字, 顏色, 字級)
    mode_text = "清單模式" if snap["mode"] == "checklist" else "依序模式"
    panel.append((10, 6, f"{mode_text}  完成 {snap['cycle_count']} 輪  漏做 {snap['bad_cycle_count']} 輪", (255, 255, 255), 24))
    y0 = 40
    for i, (s, state) in enumerate(zip(project.steps, snap["step_states"])):
        if snap["mode"] == "checklist":
            acc = fsm.acc_sec.get(s.step_index, 0.0)
            mark = "●" if state == "done" else "○"
            extra = "" if state == "done" else f"  {acc:.1f}/{s.dwell_time_sec:.1f}s"
            req = "" if s.required else "（選配）"
        else:
            mark = "●" if state == "done" else "▶" if state == "active" else "○"
            extra = f"  {snap['progress'] * 100:.0f}%" if state == "active" else ""
            req = ""
        panel.append((14, y0 + 30 * i, f"{mark} {s.step_name}{req}{extra}", colors[state], 22))
    y_next = y0 + 30 * len(project.steps)
    if ctx.gesture_probs:
        top = max(ctx.gesture_probs, key=ctx.gesture_probs.get)
        panel.append((14, y_next, f"手勢：{top} {ctx.gesture_probs[top]:.2f}", (255, 255, 0), 20))
        y_next += 30
    if state_runner is not None and state_runner.enabled:
        r = state_runner.stream.last
        if state_runner.lib.examples:
            txt = f"盒內狀態：{ctx.box_state or '？'}" + (f"（{r.confidence:.2f}）" if r.state else "") + (f" {r.reason}" if r.reason else "")
            panel.append((14, y_next, txt, (0, 255, 255), 20)); y_next += 30
        if state_runner.items is not None:
            if state_runner.open_closed is not None and state_runner.region is not None:
                oc = {True: "開著", False: "蓋著（視同盒子不在）", None: "？"}[state_runner.box_open]
                panel.append((14, y_next, f"盒子：{oc}", (0, 255, 0) if state_runner.box_open else (0, 200, 255), 20)); y_next += 30
            items = "、".join(f"{n} {c:.2f}" for n, c in sorted(state_runner.items_present.items())) or "（沒看到任何物品）"
            head = "桌上看到" if state_runner.fixed_items else "盒內物品"
            panel.append((14, y_next, f"{head}：{items}" + (f" {r.reason}" if r.reason and not state_runner.lib.examples else ""), (0, 255, 0), 20)); y_next += 30
            if state_runner.region is not None:
                rx, ry = state_runner.region[0], state_runner.region[1]
                for h in state_runner.item_hits:
                    x, y, w, hh = h.rect
                    cv2.rectangle(frame, (S(rx + x), S(ry + y)), (S(rx + x + w), S(ry + y + hh)), (0, 255, 0), 3)
                    texts.append((S(rx + x) + 4, S(ry + y) + 24, f"{h.name} {h.conf:.2f}", (0, 255, 0), 18))
        if state_runner.region is not None:
            x, y, w, h = state_runner.region
            # 黃＝盒子現在看得到；灰＝超過 1 秒沒偵測到、框是沿用上次的位置（這時不判，只是畫出來）
            fresh = state_runner.last_age is None or state_runner.last_age <= STATE_FRESH_SEC
            col = (0, 255, 255) if fresh else (160, 160, 160)
            cv2.rectangle(frame, (S(x), S(y)), (S(x + w), S(y + h)), col, 2)
            fixed = state_runner.fixed_items and state_runner.items is not None
            label = "偵測範圍（整張桌面）" if fixed else "盒內狀態區" if fresh else "盒內狀態區（沿用上次位置）"
            texts.append((S(x) + 4, S(y + h) - 26, label, col, 18))
    # 盒子拿走後要「錨點過期（max_age_sec）＋結束條件維持 hold 秒」才結算，使用者看起來像沒反應 → 把倒數畫出來
    if fsm.mode == "checklist" and project.cycle_end is not None and fsm.cycle_active:
        age = state_runner.last_age if state_runner is not None else None
        box_spec = next((a for a in project.anchors if a.name == project.state_anchor), None)
        if fsm.end_hold_sec > 0:
            panel.append((14, y_next, f"結束條件成立 {fsm.end_hold_sec:.1f}/{project.cycle_end.dwell_time_sec:.1f} 秒後結算這一輪", (0, 200, 255), 20)); y_next += 30
        elif project.cycle_end.type == "object_absent" and age is not None and age > STATE_FRESH_SEC and box_spec is not None:
            panel.append((14, y_next, f"盒子 {age:.1f} 秒沒看到（{box_spec.max_age_sec:.0f} 秒後視為拿走）", (0, 200, 255), 20)); y_next += 30
    if not snap["armed"] and not snap["alert_active"]:
        panel.append((14, y_next, "本輪已結束，等待下一輪開始", (0, 200, 255), 20)); y_next += 30
    panel_h = y_next + 6
    top = max(0, fh - panel_h)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, top), (fw, fh), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    texts.extend((x, top + y, t, c, sz) for x, y, t, c, sz in panel)
    if snap["alert_active"]:
        cv2.rectangle(frame, (0, 0), (fw - 1, fh - 1), (0, 0, 255), 16)
        texts.append((30, fh // 2 - 30, snap["alert_message"], (0, 0, 255), 44))
    texts.append((10, 6, fps_text, (255, 255, 255), 18))        # 面板搬到下方後，fps 改放左上角
    return draw_texts(frame, texts)


def _main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="SOP-Guard 核心引擎開發測試入口")
    parser.add_argument("--project", required=True, help="專案資料夾路徑或 sop_projects 下的專案名稱")
    parser.add_argument("--source", required=True, help="相機編號（整數）或影片檔路徑")
    parser.add_argument("--show", action="store_true", help="開 OpenCV 視窗顯示（只畫 ASCII）")
    parser.add_argument("--max-frames", type=int, default=0, help="最多處理幾格，0 = 全部")
    parser.add_argument("--auto-reset-sec", type=float, default=2.0, help="警告後幾秒自動 reset（開發用）")
    parser.add_argument("--model", default=os.path.join(app_base_dir(), "models", "hand_landmarker.task"))
    parser.add_argument("--yolo", default=os.path.join(app_base_dir(), "models", "yolov8s-worldv2.pt"), help="YOLO-World 權重（專案有 anchors 時需要）")
    parser.add_argument("--write-logs", action="store_true", help="把事件與截圖寫進專案的 error_logs/")
    parser.add_argument("--no-pace", action="store_true", help="影片檔來源 + --show 時不照原速播放（預設照 fps 放慢）")
    parser.add_argument("--save-preview", default=None, help="每 150 格把預覽畫面存一張 jpg 到此資料夾（不開視窗也可用）")
    parser.add_argument("--sync-yolo", action="store_true", help="相機來源也用同步偵測（預設相機走背景執行緒、影片檔走同步以便重現）")
    parser.add_argument("--yolo-threads", type=int, default=2, help="YOLO 用幾個 CPU 執行緒（留一些給 MediaPipe）")
    parser.add_argument("--stats-log", default=None, help="每 60 秒把 FPS／記憶體／輪數寫一行到這個 csv（長時間測試用）")
    parser.add_argument("--max-seconds", type=float, default=0, help="跑這麼多秒就停，0 = 不限")
    parser.add_argument("--state-log", default=None, help="盒內狀態每次判定寫一列到這個 csv（調範例／規則用）")
    args = parser.parse_args(argv)

    project = load_project(args.project)
    log_dir = os.path.join(project.dir_path, "error_logs") if args.write_logs else None
    fsm = SOPStateMachine(project, error_log_dir=log_dir, sound_enabled=False)
    tracker = HandTracker(args.model)
    scene = SceneTracker(project.anchors, tuple(project.work_area) if project.work_area else None)
    is_camera = is_camera_source(args.source)
    detector = None; async_det = None
    if project.anchors:
        from object_detector import AsyncDetector, ObjectDetector
        detector = ObjectDetector(args.yolo, [a.prompt for a in project.anchors], threads=args.yolo_threads)
        if is_camera and not args.sync_yolo:
            async_det = AsyncDetector(detector)
    gesture = None
    if project.gesture_model:
        from gesture_classifier import GestureClassifier, GestureStream, hand_frame_features
        gesture = GestureStream(GestureClassifier.load(os.path.join(project.dir_path, project.gesture_model)))
    state_runner = BoxStateRunner(project)
    limit_torch_threads()                     # 模型都載完再設（各模型載入時會自己設執行緒數）
    if not state_runner.enabled:
        state_runner = None

    if is_camera:
        source = CameraSource(args.source.strip(), project.resolution[0], project.resolution[1])
        print(f"相機 {args.source}：{source.resolution[0]}x{source.resolution[1]} @ {source.fps:.1f} fps")
    else:
        source = VideoFileSource(args.source)
    if tuple(source.resolution) != tuple(project.resolution):
        print(f"注意：來源解析度 {source.resolution[0]}x{source.resolution[1]} 與專案 resolution "
              f"{project.resolution[0]}x{project.resolution[1]} 不同，固定座標／work_area 可能對不上")

    stats_f = None
    if args.stats_log:
        os.makedirs(os.path.dirname(os.path.abspath(args.stats_log)), exist_ok=True)
        is_new = not os.path.exists(args.stats_log)
        stats_f = open(args.stats_log, "a", encoding="utf-8-sig" if is_new else "utf-8", newline="")
        if is_new:
            stats_f.write("time_iso,elapsed_sec,display_fps,detect_fps,object_ms,rss_mb,cycles,bad_cycles,alerts\n")
    next_stats = 60.0
    state_f = None
    if args.state_log and state_runner is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.state_log)), exist_ok=True)
        state_f = open(args.state_log, "w", encoding="utf-8-sig", newline="")
        state_f.write("frame,t,why,region,hands,raw,conf,margin,current,age,items\n")

    frames = 0; detections = 0; detect_time = 0.0; obj_time = 0.0; obj_runs = 0; printed = 0; last_hand_t = -1.0; last_obj_t = -1e9
    obj_interval = max(1, project.detect_every_n_frames) / float(project.fps or 30.0)
    wall0 = time.perf_counter()
    ctx = FrameContext()
    try:
        while True:
            frame = source.read()
            if frame is None:
                break
            t_now = source.time_sec
            # 手部每 DETECT_EVERY_N_FRAMES 格一次（影片檔也一樣；原本影片每格都做，手部一格 57 ms 佔掉將近一半），
            # 沒有規則用到手的專案整個跳過
            if not project.needs_hands:
                points = []
            elif frames % DETECT_EVERY_N_FRAMES and t_now - last_hand_t < HAND_MIN_INTERVAL_SEC:
                points = tracker.last_points
            else:
                last_hand_t = t_now
                t0 = time.perf_counter()
                points = tracker.detect(frame, int(t_now * 1000))
                detect_time += time.perf_counter() - t0
                detections += 1
            if async_det is not None:
                if not async_det.busy() and t_now - last_obj_t >= obj_interval:   # 每 detect_every_n_frames÷fps 秒送一次，不要有空就跑
                    async_det.submit(frame, t_now); last_obj_t = t_now
                res = async_det.poll()
                if res is not None:
                    scene.update_detections(res[0], res[1])
                    obj_runs = async_det.runs; obj_time = async_det.total_ms / 1000
                if async_det.error is not None:
                    raise async_det.error
            elif detector is not None and frames % project.detect_every_n_frames == 0:
                t0 = time.perf_counter()
                scene.update_detections(detector.detect(frame), t_now)
                obj_time += time.perf_counter() - t0; obj_runs += 1
            anchors = scene.anchors_at(t_now)
            probs = {}
            if gesture is not None:
                refs = [union_rect(anchors.get(name, [])) for name in project.gesture_anchors]
                probs = dict(gesture.push(hand_frame_features(tracker.last_hands_xy, refs, (frame.shape[1], frame.shape[0]))))
            box_state = None
            if state_runner is not None:
                age = scene.age(project.state_anchor, t_now) if project.state_anchor else None
                box_state = state_runner.update(frame, anchors, points, anchor_age=age, t_now=t_now)
                if state_f is not None and state_runner.debug is not None:
                    d = state_runner.debug; reg = state_runner.region
                    state_f.write(f"{frames},{t_now:.2f},{d['why']},{'' if reg is None else ' '.join(str(int(v)) for v in reg)},"
                                  f"{len(points)},{d.get('raw', '') or ''},{d.get('conf', '')},{d.get('margin', '')},{box_state or ''},"
                                  f"{'' if age is None else f'{age:.2f}'},{d.get('items', '')}\n")
            ctx = FrameContext(hand_points=points, hands_xy=tracker.last_hands_xy, anchors=anchors, gesture_probs=probs, box_state=box_state,
                               items_present=set(state_runner.items_present) if state_runner is not None else set(),
                               box_open=state_runner.box_open if state_runner is not None else None,
                               box_seen=state_runner.box_seen if state_runner is not None else False)
            fsm.update(frame, ctx, t_now)
            if fsm.alert_active and (t_now - fsm.alert_started_at) >= args.auto_reset_sec:
                fsm.reset()
            while printed < len(fsm.events):
                ev = fsm.events[printed]
                printed += 1
                idx = f" {ev['step_index']}" if ev["step_index"] != "" else ""
                print(f"[t={ev['t']:.2f}s] {ev['event_type']}{idx} {ev['message']}".rstrip())
            frames += 1
            if args.show or (args.save_preview and frames % 150 == 0):
                elapsed = time.perf_counter() - wall0
                vis = _draw_dev_overlay(frame, project, fsm, ctx, f"處理 {frames / elapsed:.1f} fps  t={t_now:.1f}s" if elapsed else "", state_runner)
                if args.save_preview and frames % 150 == 0:
                    os.makedirs(args.save_preview, exist_ok=True)
                    write_image(os.path.join(args.save_preview, f"preview_{frames:06d}.jpg"), vis)
                if args.show:
                    cv2.imshow("SOP-Guard 預覽（按 q 離開）", vis)
                    wait_ms = 1
                    if not is_camera and not args.no_pace:
                        due = wall0 + t_now - time.perf_counter()
                        wait_ms = max(1, int(due * 1000))
                    if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                        break
            elapsed = time.perf_counter() - wall0
            if stats_f is not None and elapsed >= next_stats:
                next_stats += 60.0
                rss = ""
                try:
                    import psutil
                    rss = f"{psutil.Process().memory_info().rss / 1048576:.0f}"
                except Exception:
                    pass
                n_alerts = sum(1 for ev in fsm.events if ev["event_type"].startswith("alert_"))
                stats_f.write(f"{datetime.now().isoformat(timespec='seconds')},{elapsed:.0f},{frames / elapsed:.1f},"
                              f"{detections / detect_time if detect_time else 0:.1f},{obj_time / obj_runs * 1000 if obj_runs else 0:.0f},"
                              f"{rss},{fsm.cycle_count},{fsm.bad_cycle_count},{n_alerts}\n")
                stats_f.flush()
            if args.max_frames and frames >= args.max_frames:
                break
            if args.max_seconds and elapsed >= args.max_seconds:
                break
    finally:
        wall = time.perf_counter() - wall0
        tracker.close()
        if async_det is not None:
            async_det.close()
        source.release()
        if stats_f is not None:
            stats_f.close()
        if state_f is not None:
            state_f.close()
        if args.show:
            cv2.destroyAllWindows()

    counts = {k: 0 for k in EVENT_TYPES}
    for ev in fsm.events:
        counts[ev["event_type"]] += 1
    detect_fps = detections / detect_time if detect_time else 0.0
    display_fps = frames / wall if wall else 0.0
    obj_ms = obj_time / obj_runs * 1000 if obj_runs else 0.0
    print(f"cycles={fsm.cycle_count} bad_cycles={fsm.bad_cycle_count} discarded={counts['cycle_discarded']} "
          f"alert_skip={counts['alert_skip']} alert_missing={counts['alert_missing']} alert_timeout={counts['alert_timeout']} "
          f"frames={frames} detect_fps={detect_fps:.1f} display_fps={display_fps:.1f} object_detect_ms={obj_ms:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
