"""監控服務（不依賴 Qt）：來源 → 手部 → 物件（背景執行緒）→ 狀態機 → 疊字畫面 → JPEG。

給網頁版（web_server.py）用；迴圈邏輯與 main_ui.MonitorWorker 相同：載入階段訊息、示範影片照原速放、
警告沒人解除時自動解除（示範影片 3 秒、相機 15 秒）。畫面只編碼一次 JPEG，多個瀏覽器共用同一份。
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime

import cv2

from core_detector import (DETECT_EVERY_N_FRAMES, HAND_MIN_INTERVAL_SEC, BoxStateRunner, FrameContext, HandTracker, SOPStateMachine,
                           _draw_dev_overlay, limit_torch_threads, union_rect)
from project_store import Project, app_base_dir, load_project
from scene import SceneTracker
from video_recorder import CameraSource, VideoFileSource, is_camera_source, write_image

HAND_MODEL = os.path.join(app_base_dir(), "models", "hand_landmarker.task")
YOLO_WEIGHTS = os.path.join(app_base_dir(), "models", "yolov8s-worldv2.pt")
CAMERA_SOURCE_FILE = os.path.join(app_base_dir(), "camera_source.txt")   # 記住的相機來源（明文；rtsp 網址含密碼，只留在這台）
JPEG_QUALITY = 80
MAX_EVENTS = 60
DISPLAY_FPS = 25          # 相機來源：顯示執行緒的上限（相機每一格疊上最近一次辨識結果就送，不等辨識跑完）


def load_camera_source() -> str:
    try:
        with open(CAMERA_SOURCE_FILE, "r", encoding="utf-8-sig") as f:
            return f.read().strip() or "0"
    except OSError:
        return "0"


class MonitorService:
    """一台電腦一個監控（一次一個專案、一個來源）。start() 立刻回來，模型在背景載入；snapshot() 隨時可叫。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._reset = threading.Event()
        self._jpeg: bytes | None = None
        self._frame_id = 0
        self._last_vis = None
        self._snap: dict = {}
        self._live = None            # 辨識執行緒最近一次的 (ctx, fsm, state_runner, 辨識 fps)，顯示執行緒拿來疊字
        self._disp_fps = 0.0
        self.project: Project | None = None
        self.source_kind = "demo"
        self.loading: str | None = None
        self.error: str | None = None
        self.finished: str | None = None

    # ---- 控制

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, project_name: str, source_kind: str = "demo", source: str | None = None) -> None:
        if self.running:
            raise ValueError("監控已在執行中，請先停止")
        if not project_name or not project_name.strip():
            from project_store import list_projects
            for n in list_projects():
                try:
                    if load_project(n).steps:
                        project_name = n
                        break
                except Exception:
                    pass
        try:
            project = load_project(project_name)
        except Exception as e:
            raise ValueError(f"載入專案失敗：{e}")
        if not project.steps:
            raise ValueError("這個專案還沒有步驟，請先在桌面程式的「② 標記物品」產生步驟")
        if source_kind == "demo":
            src = project.video_path
            if not os.path.isfile(src):
                raise ValueError("這個專案沒有示範影片")
        elif source_kind == "camera":
            src = (source or load_camera_source()).strip()
        else:
            raise ValueError("source_kind 只能是 demo 或 camera")
        with self._lock:
            self.project = project; self.source_kind = source_kind
            self.loading = "載入中（第一次約 10～30 秒，請等畫面出現）…"; self.error = None; self.finished = None
            self._snap = {}; self._jpeg = None; self._last_vis = None
        self._stop.clear(); self._reset.clear()
        self._thread = threading.Thread(target=self._run, args=(project, src, is_camera_source(src)), daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout)
        self._thread = None
        with self._lock:
            self.loading = None

    def request_reset(self) -> None:
        self._reset.set()

    # ---- 給網頁的資料

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def wait_frame(self, last_id: int, timeout: float = 1.0) -> tuple[int, bytes | None]:
        """等到比 last_id 新的畫面（串流用）；逾時回傳目前的。"""
        with self._cond:
            if self._frame_id <= last_id:
                self._cond.wait(timeout)
            return self._frame_id, self._jpeg

    def snapshot(self) -> dict:
        with self._lock:
            d = {"running": self.running, "loading": self.loading, "error": self.error, "finished": self.finished,
                 "project": self.project.project_name if self.project else None, "source_kind": self.source_kind,
                 "frame_id": self._frame_id}
            d.update(self._snap)
        return d

    def save_screenshot(self) -> str | None:
        with self._lock:
            vis = self._last_vis
        if vis is None:
            return None
        out = os.path.join(app_base_dir(), "cache", f"screenshot_{datetime.now():%Y%m%d_%H%M%S}.jpg")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        write_image(out, vis)
        return out

    # ---- 迴圈

    def _set_loading(self, text: str | None) -> None:
        with self._lock:
            self.loading = text

    def _publish_frame(self, vis) -> None:
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        with self._cond:
            if ok:
                self._jpeg = buf.tobytes(); self._frame_id += 1
            self._last_vis = vis
            self.loading = None
            self._cond.notify_all()

    def _publish_state(self, fsm: SOPStateMachine, frames: int, fps: float) -> None:
        snap = fsm.snapshot()
        steps = []
        for s, state in zip(fsm.steps, snap["step_states"]):
            steps.append({"index": s.step_index, "name": s.step_name, "state": state, "required": s.required,
                          "acc": round(fsm.acc_sec.get(s.step_index, 0.0), 2) if fsm.mode == "checklist" else None,
                          "dwell": s.dwell_time_sec})
        events = [{"t": round(e.get("t", 0.0), 2), "type": e.get("event_type", ""), "message": e.get("message", ""),
                   "step_index": e.get("step_index", "")} for e in fsm.events[-MAX_EVENTS:]]
        with self._lock:
            self._snap = {"mode": snap["mode"], "cycle_count": snap["cycle_count"], "bad_cycle_count": snap["bad_cycle_count"],
                          "alert_active": snap["alert_active"], "alert_message": snap["alert_message"], "armed": snap["armed"],
                          "steps": steps, "events": events, "frames": frames, "fps": round(fps, 1), "display_fps": round(self._disp_fps, 1),
                          "n_events": len(fsm.events)}

    def _display_loop(self, p: Project, src) -> None:
        """相機來源：相機每一格都疊上最近一次辨識結果、編碼、送出（上限 DISPLAY_FPS）。辨識執行緒另外跑自己的速度，
        畫面不再等它。疊字面板的文字有快取，一格約 5 ms。"""
        last_seq = -1; n = 0; t0 = time.perf_counter(); interval = 1.0 / DISPLAY_FPS
        while not self._stop.is_set():
            seq, frame = src.latest()
            if frame is None or seq == last_seq:
                time.sleep(0.004); continue
            with self._lock:
                live = self._live
            if live is None:
                time.sleep(0.02); continue
            last_seq = seq; tick = time.perf_counter()
            ctx, fsm, runner, rec_fps = live
            vis = _draw_dev_overlay(frame, p, fsm, ctx, f"辨識 {rec_fps:.1f} fps  畫面 {self._disp_fps:.1f} fps", runner)
            self._publish_frame(vis)
            n += 1; el = time.perf_counter() - t0
            if el >= 1.0:
                self._disp_fps = n / el; n = 0; t0 = time.perf_counter()
            due = interval - (time.perf_counter() - tick)
            if due > 0:
                time.sleep(due)

    def _run(self, p: Project, source: str, is_camera: bool) -> None:
        tracker = None; async_det = None; src = None
        try:
            fsm = SOPStateMachine(p, error_log_dir=os.path.join(p.dir_path, "error_logs"), sound_enabled=False)
            self._set_loading("載入中 1/4：手部模型…")
            tracker = HandTracker(HAND_MODEL)
            scene = SceneTracker(p.anchors, tuple(p.work_area) if p.work_area else None)
            if p.anchors:
                self._set_loading("載入中 2/4：物件偵測模型（YOLO）…")
                import numpy as np
                from object_detector import AsyncDetector, ObjectDetector
                obj_det = ObjectDetector(YOLO_WEIGHTS, [a.prompt for a in p.anchors])
                if getattr(obj_det, "device", "cpu") != "cpu":
                    obj_det.detect(np.zeros((64, 64, 3), dtype=np.uint8))
                async_det = AsyncDetector(obj_det)
            gesture = None
            if p.gesture_model:
                from gesture_classifier import GestureClassifier, GestureStream, hand_frame_features
                gesture = GestureStream(GestureClassifier.load(os.path.join(p.dir_path, p.gesture_model)))
            self._set_loading("載入中 3/4：物品／狀態辨識模型…")
            state_runner = BoxStateRunner(p)
            limit_torch_threads()                     # 監控時 PyTorch 只用 1 個執行緒，減少與手部／物品模型搶 CPU
            if not state_runner.enabled:
                state_runner = None
            else:
                state_runner.lib._ensure_vecs()
                if getattr(state_runner.items, "backend", "") == "cuda":
                    import numpy as np
                    state_runner.items.detect(np.zeros((64, 64, 3), dtype=np.uint8))
            self._set_loading("載入中 4/4：開啟" + ("相機" if is_camera else "示範影片") + "…")
            src = CameraSource(source, p.resolution[0], p.resolution[1]) if is_camera else VideoFileSource(source)
        except Exception as e:
            with self._lock:
                self.error = f"{type(e).__name__}: {e}"; self.loading = None
            self._cleanup(tracker, async_det, src); return
        frames = 0; wall0 = time.perf_counter(); disp = None; last_hand_t = -1.0; last_obj_t = -1e9
        obj_interval = max(1, p.detect_every_n_frames) / float(p.fps or 30.0)
        if is_camera:
            disp = threading.Thread(target=self._display_loop, args=(p, src), daemon=True); disp.start()
        try:
            while not self._stop.is_set():
                frame = src.read()
                if frame is None:
                    break
                t_now = src.time_sec
                if not is_camera:
                    # 示範影片照原速播：處理跟不上（1080×1920 的現場影片這台只有 5～7 fps）就跳格，每處理 1 格最多跳 2 格；
                    # 狀態機用影片時間算 dt，跳格不影響停留秒數。相機來源本來就只拿最新一格，不用跳
                    skipped = 0
                    while skipped < 2 and (time.perf_counter() - wall0) - src.time_sec > 0.15:
                        nxt = src.read()
                        if nxt is None:
                            break
                        frame = nxt; t_now = src.time_sec; skipped += 1
                if not p.needs_hands:                        # 沒有規則用到手的專案跳過 MediaPipe（每格 50～60 ms）
                    points = []
                elif frames % DETECT_EVERY_N_FRAMES and t_now - last_hand_t < HAND_MIN_INTERVAL_SEC:
                    points = tracker.last_points             # 處理夠快才隔格做；慢的時候每格都做，取樣才不會稀
                else:
                    last_hand_t = t_now
                    points = tracker.detect(frame, int(t_now * 1000))
                if async_det is not None:
                    # 物件偵測（YOLO-World）不再「有空就跑」：每 detect_every_n_frames÷fps 秒送一次（現場專案 15 格＝0.5 秒）。
                    # 不停地跑每次 300 ms 等於吃掉一整顆核心，手部和物品模型只剩 4.7 fps；錨點本來就有 1～2.5 秒沿用
                    if not async_det.busy() and t_now - last_obj_t >= obj_interval:
                        async_det.submit(frame, t_now); last_obj_t = t_now
                    r = async_det.poll()
                    if r is not None:
                        scene.update_detections(r[0], r[1])
                    if async_det.error is not None:
                        raise async_det.error
                anchors = scene.anchors_at(t_now)
                probs = {}
                if gesture is not None:
                    refs = [union_rect(anchors.get(n, [])) for n in p.gesture_anchors]
                    probs = dict(gesture.push(hand_frame_features(tracker.last_hands_xy, refs, (frame.shape[1], frame.shape[0]))))
                box_state = None
                if state_runner is not None:
                    age = scene.age(p.state_anchor, t_now) if p.state_anchor else None
                    box_state = state_runner.update(frame, anchors, points, anchor_age=age, t_now=t_now)
                ctx = FrameContext(hand_points=points, hands_xy=tracker.last_hands_xy, anchors=anchors, gesture_probs=probs, box_state=box_state,
                                   items_present=set(state_runner.items_present) if state_runner is not None else set(),
                                   box_open=state_runner.box_open if state_runner is not None else None,
                                   box_seen=state_runner.box_seen if state_runner is not None else False)
                fsm.update(frame, ctx, t_now)
                auto_clear = 3.0 if not is_camera else 15.0
                if fsm.alert_active and (t_now - fsm.alert_started_at) >= auto_clear:
                    self._reset.set()
                if self._reset.is_set():
                    self._reset.clear(); fsm.reset()
                    if state_runner is not None:
                        state_runner.stream.reset()
                frames += 1
                if frames == 1:
                    wall0 = time.perf_counter() - max(t_now, 1.0 / float(p.fps or 30.0))
                el = time.perf_counter() - wall0; rec_fps = frames / el if el else 0.0
                with self._lock:
                    self._live = (ctx, fsm, state_runner, rec_fps)
                self._publish_state(fsm, frames, rec_fps)
                if disp is None:                             # 影片檔：沒有顯示執行緒，處理完的格直接送
                    self._publish_frame(_draw_dev_overlay(frame, p, fsm, ctx, f"{rec_fps:.1f} fps", state_runner))
                if not is_camera:
                    due = wall0 + t_now - time.perf_counter()
                    if due > 0:
                        time.sleep(min(due, 0.1))
            with self._lock:
                if frames == 0 and not self._stop.is_set():
                    self.error = "來源沒有送出任何畫面（相機連不上，或示範影片讀不到）"
                elif not is_camera and not self._stop.is_set():
                    self.finished = f"示範影片播完：完成 {fsm.cycle_count} 輪、漏做 {fsm.bad_cycle_count} 輪"
                self.loading = None
        except Exception as e:
            with self._lock:
                self.error = f"{type(e).__name__}: {e}"; self.loading = None
        finally:
            self._stop.set()
            if disp is not None:
                disp.join(2.0)
            with self._lock:
                self._live = None
            self._cleanup(tracker, async_det, src)

    @staticmethod
    def _cleanup(tracker, async_det, src) -> None:
        for obj, fn in ((tracker, "close"), (async_det, "close"), (src, "release")):
            if obj is not None:
                try:
                    getattr(obj, fn)()
                except Exception:
                    pass
