"""SOP-Guard 操作介面（PySide6）。操作者只做三件事：① 錄一段示範 ② 在時間軸上分段說明動作 ③ 訓練後監控。

    python main_ui.py

左側：專案清單。右側三個分頁：
  ① 錄製：相機預覽、開始／停止錄製（存成專案的 demo.avi）、匯入影片。
  ② 分段說明：拉時間軸、設起點／終點、輸入動作名稱、加入分段；「訓練並產生步驟」一鍵完成。
  ③ 監控：啟動／停止、重置／解除警告、輪數統計、事件清單；畫面上的框跟著偵測到的物件動。
工程師要設定物件錨點、區域、物件型結束條件時，直接改 sop_projects/<專案>/config.json。
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime

# PyTorch 的 DLL 必須在 Qt 之前載入，否則之後用到物件偵測時會 WinError 1114（c10.dll 初始化失敗）。
# 沒裝 torch 的環境（只用手勢、不用物件錨點）也要能跑，所以失敗就略過。
try:
    import torch  # noqa: F401
except Exception:  # pragma: no cover
    torch = None

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QProcess, QProcessEnvironment, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QPolygon, QShortcut
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QInputDialog, QLabel,
                               QLineEdit, QListWidget, QMainWindow, QMessageBox, QProgressBar, QPushButton, QSlider,
                               QStackedWidget, QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit, QVBoxLayout, QWidget)

import training
from core_detector import (DETECT_EVERY_N_FRAMES, HAND_MIN_INTERVAL_SEC, BoxStateRunner, FrameContext, HandTracker, SOPStateMachine,
                           _draw_dev_overlay, limit_torch_threads, state_region, union_rect)
from scene import Detection, SceneTracker as _SceneTracker
from state_recognizer import StateLibrary
from project_store import (Project, ProjectConfigError, app_base_dir, delete_project, list_projects, load_project,
                           project_to_dict, projects_dir, validate_project_name)
from scene import SceneTracker
from video_recorder import CameraSource, FrameReader, Recorder, VideoFileSource, import_demo_video, is_camera_source, write_image
import item_recognizer as items

HAND_MODEL = os.path.join(app_base_dir(), "models", "hand_landmarker.task")
YOLO_WEIGHTS = os.path.join(app_base_dir(), "models", "yolov8s-worldv2.pt")
CAMERA_SOURCE_FILE = os.path.join(app_base_dir(), "camera_source.txt")   # 記住的相機來源（明文；rtsp 網址含密碼）
BIG_BTN = "QPushButton { font-size: 18px; padding: 10px 18px; }"


def load_camera_source() -> str:
    try:
        with open(CAMERA_SOURCE_FILE, "r", encoding="utf-8-sig") as f:
            s = f.read().strip()
            return s or "0"
    except OSError:
        return "0"


def save_camera_source(source: str) -> None:
    with open(CAMERA_SOURCE_FILE, "w", encoding="utf-8") as f:
        f.write(source.strip() + "\n")


def create_project_with_defaults(name: str, width: int, height: int, fps: float) -> str:
    """建一個操作者用的 schema 2 專案：沒有錨點、步驟由訓練產生、一輪結束＝手離開畫面 3 秒。回傳專案資料夾。"""
    validate_project_name(name)
    d = os.path.join(projects_dir(), name)
    if os.path.exists(d):
        raise ProjectConfigError(f"專案「{name}」已存在")
    cfg = {"schema_version": 2, "project_name": name, "video_file": "demo.avi", "resolution": [int(width), int(height)],
           "fps": round(float(fps), 2), "skip_alert_sec": 1.0, "cycle_arm_clear_sec": 0.5, "reset_zone": None,
           "reset_dwell_sec": 1.0, "work_area": None, "detect_every_n_frames": 15, "gesture_model": None,
           "gesture_anchors": [], "cycle_end": {"type": "hands_absent", "hold_sec": 3.0}, "cycle_timeout_sec": 120.0,
           "cycle_min_sec": 5.0, "anchors": [], "zones": [], "steps": []}
    os.makedirs(os.path.join(d, "gestures"), exist_ok=True); os.makedirs(os.path.join(d, "error_logs"), exist_ok=True)
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return d


def label_color(label: str) -> QColor:
    """同一個動作固定同一個顏色（用名稱算色相）。"""
    h = sum(ord(c) * (i + 7) for i, c in enumerate(label)) % 360
    return QColor.fromHsv(h, 170, 220)


class TimelineBar(QWidget):
    """分段時間軸：已標的分段畫成色塊，起點／終點畫旗標，目前位置畫白線；點一下可跳到該位置。"""
    seek = Signal(int)

    def __init__(self):
        super().__init__()
        self.total = 0; self.segments: list[dict] = []; self.proposals: list[dict] = []
        self.start = None; self.end = None; self.current = 0
        self.setFixedHeight(44); self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("色塊＝已標的分段；淡色斜線塊＝系統建議、待確認；綠旗＝起點；紅旗＝終點；白線＝目前位置。點一下可跳到該處。")

    def set_state(self, total: int, segments: list[dict], start, end, current: int, proposals: list[dict] | None = None) -> None:
        self.total, self.segments, self.start, self.end, self.current = max(total, 1), list(segments), start, end, current
        self.proposals = list(proposals or [])
        self.update()

    def _x(self, frame: int) -> int:
        return int(frame / max(self.total - 1, 1) * (self.width() - 1))

    def _frame_at(self, x: int) -> int:
        return int(round(x / max(self.width() - 1, 1) * (self.total - 1)))

    def paintEvent(self, event) -> None:
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(45, 45, 45))
        band_top, band_h = 12, h - 16
        for s in self.proposals:   # 建議：淡色＋斜線，與已標的分開
            x0, x1 = self._x(int(s["start"])), self._x(int(s["end"]) - 1)
            c = label_color(s["label"]); c.setAlpha(110)
            p.fillRect(x0, band_top, max(x1 - x0, 2), band_h, c)
            p.setPen(QPen(QColor(255, 255, 255, 120), 1, Qt.DashLine)); p.drawRect(x0, band_top, max(x1 - x0, 2), band_h)
            if x1 - x0 > 28:
                p.setPen(QColor(30, 30, 30)); p.drawText(x0 + 4, band_top, x1 - x0 - 6, band_h, Qt.AlignVCenter | Qt.AlignLeft, s["label"] + "?")
        for s in self.segments:
            x0, x1 = self._x(int(s["start"])), self._x(int(s["end"]) - 1)
            c = label_color(s["label"])
            p.fillRect(x0, band_top, max(x1 - x0, 2), band_h, c)
            p.setPen(QPen(QColor(20, 20, 20), 1)); p.drawRect(x0, band_top, max(x1 - x0, 2), band_h)
            if x1 - x0 > 28:
                p.setPen(QColor(0, 0, 0)); p.drawText(x0 + 4, band_top, x1 - x0 - 6, band_h, Qt.AlignVCenter | Qt.AlignLeft, s["label"])
        if self.start is not None and self.end is not None:
            a, b = sorted((self.start, self.end))
            p.fillRect(self._x(a), band_top, max(self._x(b) - self._x(a), 2), band_h, QColor(255, 255, 255, 70))
        for frame, color in ((self.start, QColor(40, 200, 80)), (self.end, QColor(230, 60, 60))):
            if frame is None:
                continue
            x = self._x(frame)
            p.setPen(QPen(color, 2)); p.drawLine(x, 0, x, h)
            p.setBrush(color); p.setPen(Qt.NoPen)
            p.drawPolygon(QPolygon([QPoint(x, 0), QPoint(x + 12, 0), QPoint(x, 11)]) if color.green() > 150
                          else QPolygon([QPoint(x, 0), QPoint(x - 12, 0), QPoint(x, 11)]))
        x = self._x(self.current)
        p.setPen(QPen(QColor(255, 255, 255), 2)); p.drawLine(x, 0, x, h)
        p.end()

    def mousePressEvent(self, event) -> None:
        self.seek.emit(self._frame_at(int(event.position().x())))

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.LeftButton:
            self.seek.emit(self._frame_at(int(event.position().x())))


def bgr_to_pixmap(frame, max_w: int, max_h: int) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(img).scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


# ---------------------------------------------------------------- 背景工作

class RecordWorker(QThread):
    """相機預覽＋錄影。"""
    frame_ready = Signal(object)
    info = Signal(str)
    failed = Signal(str)

    def __init__(self, cam_id, width: int, height: int):
        super().__init__()
        self.cam_id, self.width, self.height = cam_id, width, height
        self._stop = threading.Event()
        self._rec = Recorder()
        self._lock = threading.Lock()
        self.resolution = (width, height); self.fps = 30.0
        self.record_path = ""

    def start_recording(self, path: str) -> None:
        with self._lock:
            self._rec.start(path, self.resolution, self.fps)
            self.record_path = path

    def stop_recording(self) -> int:
        with self._lock:
            n = self._rec.frame_count
            self._rec.stop()
        return n

    @property
    def recording(self) -> bool:
        return self._rec.recording

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            cam = CameraSource(self.cam_id, self.width, self.height, latest_only=False)   # 錄影要每一格都留，不能丟格
        except Exception as e:
            self.failed.emit(str(e)); return
        self.resolution, self.fps = cam.resolution, cam.fps
        self.info.emit(f"相機 {self.cam_id}：{cam.resolution[0]}x{cam.resolution[1]} @ {cam.fps:.1f} fps")
        try:
            while not self._stop.is_set():
                frame = cam.read()
                if frame is None:
                    self.failed.emit("相機斷線"); break
                with self._lock:
                    if self._rec.recording:
                        self._rec.write(frame)
                self.frame_ready.emit(frame)
        finally:
            with self._lock:
                if self._rec.recording:
                    self._rec.stop()
            cam.release()


class TrainWorker(QThread):
    progress = Signal(str, int, int)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, project_dir: str, min_clips: int = 3):
        super().__init__(); self.project_dir = project_dir; self.min_clips = min_clips

    def run(self) -> None:
        try:
            r = training.train_project(self.project_dir, HAND_MODEL, YOLO_WEIGHTS if os.path.isfile(YOLO_WEIGHTS) else None,
                                       progress=lambda st, d, t: self.progress.emit(st, d, t), min_clips=self.min_clips)
            self.done.emit(r)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class ItemDataWorker(QThread):
    """物件版：整支影片找盒子＋手（有快取就略過）→ 用框選的物品延伸成訓練資料。"""
    progress = Signal(str, int, int)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, project: Project, seeds):
        super().__init__(); self.project, self.seeds = project, seeds

    def run(self) -> None:
        try:
            trace = items.build_trace(self.project, HAND_MODEL, YOLO_WEIGHTS,
                                      progress=lambda a, b: self.progress.emit(f"找盒子與手：{a}/{b} 格", a, b))
            stats = items.build_dataset(self.project, trace, self.seeds,
                                        progress=lambda a, b: self.progress.emit(f"延伸框選：第 {a}/{b} 次擺放", a, b))
            self.done.emit(stats)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class ProposeWorker(QThread):
    progress = Signal(str, int, int)
    done = Signal(list)
    failed = Signal(str)

    def __init__(self, project_dir: str):
        super().__init__(); self.project_dir = project_dir

    def run(self) -> None:
        try:
            props = training.propose_for_project(self.project_dir, HAND_MODEL, YOLO_WEIGHTS if os.path.isfile(YOLO_WEIGHTS) else None,
                                                 progress=lambda st, d, t: self.progress.emit(st, d, t))
            self.done.emit(props)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class MonitorWorker(QThread):
    """來源 → 手部 → 物件（背景執行緒）→ 手勢 → 狀態機 → 畫面。"""
    frame_ready = Signal(object, object)   # (overlay_bgr, snapshot)
    events = Signal(list)
    failed = Signal(str)
    info = Signal(str)                     # 載入進度（第一張畫面前使用者才看得到有在動）

    def __init__(self, project: Project, source: str):
        super().__init__()
        self.project, self.source = project, source
        self._stop = threading.Event(); self._reset = threading.Event()
        self.frames_sent = 0

    def request_reset(self) -> None:
        self._reset.set()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        p = self.project
        try:
            fsm = SOPStateMachine(p, error_log_dir=os.path.join(p.dir_path, "error_logs"), sound_enabled=True)
            self.info.emit("載入中 1/4：手部模型…")
            tracker = HandTracker(HAND_MODEL)
            scene = SceneTracker(p.anchors, tuple(p.work_area) if p.work_area else None)
            async_det = None
            if p.anchors:
                self.info.emit("載入中 2/4：物件偵測模型（YOLO）…")
                from object_detector import AsyncDetector, ObjectDetector
                async_det = AsyncDetector(ObjectDetector(YOLO_WEIGHTS, [a.prompt for a in p.anchors]))
            gesture = None
            if p.gesture_model:
                from gesture_classifier import GestureClassifier, GestureStream, hand_frame_features
                gesture = GestureStream(GestureClassifier.load(os.path.join(p.dir_path, p.gesture_model)))
            self.info.emit("載入中 3/4：盒內狀態辨識（CLIP）…")
            state_runner = BoxStateRunner(p)
            limit_torch_threads()                     # 監控時 PyTorch 只用 1 個執行緒，減少與手部／物品模型搶 CPU
            if not state_runner.enabled:
                state_runner = None
            else:
                state_runner.lib._ensure_vecs()          # 範例向量先算好，第一張畫面才不會再卡一次
            is_camera = is_camera_source(self.source)
            self.info.emit("載入中 4/4：開啟" + ("相機" if is_camera else "示範影片") + "…")
            src = CameraSource(self.source.strip(), p.resolution[0], p.resolution[1]) if is_camera else VideoFileSource(self.source)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}"); return
        frames = 0; printed = 0; wall0 = time.perf_counter(); last_hand_t = -1.0; last_obj_t = -1e9
        obj_interval = max(1, p.detect_every_n_frames) / float(p.fps or 30.0)
        try:
            while not self._stop.is_set():
                frame = src.read()
                if frame is None:
                    break
                t_now = src.time_sec
                if not p.needs_hands:                        # 沒有規則用到手的專案跳過 MediaPipe（每格 50～60 ms）
                    points = []
                elif frames % DETECT_EVERY_N_FRAMES and t_now - last_hand_t < HAND_MIN_INTERVAL_SEC:
                    points = tracker.last_points             # 處理夠快才隔格做；慢的時候每格都做，取樣才不會稀
                else:
                    last_hand_t = t_now
                    points = tracker.detect(frame, int(t_now * 1000))
                if async_det is not None:
                    if not async_det.busy() and t_now - last_obj_t >= obj_interval:   # 每 detect_every_n_frames÷fps 秒送一次，不要有空就跑
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
                # 警告沒人解除會卡住整個狀態機（之後的輪次全部不算）：示範影片 3 秒、相機 15 秒後自動解除並記一筆「重置」
                auto_clear = 3.0 if not is_camera else 15.0
                if fsm.alert_active and (t_now - fsm.alert_started_at) >= auto_clear:
                    self._reset.set()
                if self._reset.is_set():
                    self._reset.clear(); fsm.reset()
                    if state_runner is not None:
                        state_runner.stream.reset()
                frames += 1
                if len(fsm.events) > printed:
                    self.events.emit(fsm.events[printed:]); printed = len(fsm.events)
                el = time.perf_counter() - wall0
                vis = _draw_dev_overlay(frame, p, fsm, ctx, f"{frames / el:.1f} fps" if el else "", state_runner)
                self.frame_ready.emit(vis, fsm.snapshot()); self.frames_sent += 1
                if not is_camera:
                    due = wall0 + t_now - time.perf_counter()
                    if due > 0:
                        time.sleep(min(due, 0.1))
            if self.frames_sent == 0 and not self._stop.is_set():
                self.failed.emit("來源沒有送出任何畫面（相機連不上，或示範影片讀不到）")
            elif not is_camera and not self._stop.is_set():
                self.info.emit(f"示範影片播完：完成 {fsm.cycle_count} 輪、漏做 {fsm.bad_cycle_count} 輪")
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
        finally:
            tracker.close()
            if async_det is not None:
                async_det.close()
            src.release()


# ---------------------------------------------------------------- 主視窗

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SOP-Guard")
        self.resize(1360, 860)
        self.project: Project | None = None
        self.reader: FrameReader | None = None
        self.segments: list[dict] = []
        self.proposals: list[dict] = []
        self.propose_worker: ProposeWorker | None = None
        self.seg_start: int | None = None
        self.seg_end: int | None = None
        self.rec_worker: RecordWorker | None = None
        self.mon_worker: MonitorWorker | None = None
        self.train_worker: TrainWorker | None = None
        self._build()
        self.refresh_projects()

    # ---- 版面

    def _build(self) -> None:
        root = QWidget(); self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        # 左：專案
        left = QVBoxLayout()
        left.addWidget(QLabel("專案"))
        self.project_list = QListWidget(); self.project_list.itemDoubleClicked.connect(lambda _: self.load_selected())
        left.addWidget(self.project_list, 1)
        for text, fn in (("新增專案", self.new_project), ("載入選取專案", self.load_selected), ("刪除專案", self.delete_selected)):
            b = QPushButton(text); b.setStyleSheet(BIG_BTN); b.clicked.connect(fn); left.addWidget(b)
        left.addWidget(QLabel("相機來源（編號 0、1… 或 rtsp:// 網址）"))
        self.cam_edit = QLineEdit(load_camera_source()); self.cam_edit.setPlaceholderText("0 或 rtsp://帳號:密碼@192.168.1.10:554/路徑")
        left.addWidget(self.cam_edit)
        b_remember = QPushButton("記住這個相機來源"); b_remember.setToolTip(f"存到 {CAMERA_SOURCE_FILE}（明文，含密碼），下次開啟自動填入")
        b_remember.clicked.connect(lambda: (save_camera_source(self._camera_source()), self.project_label.setText(self.project_label.text() + "\n（相機來源已記住）")))
        left.addWidget(b_remember)
        self.project_label = QLabel("（尚未載入專案）"); self.project_label.setWordWrap(True); left.addWidget(self.project_label)
        lw = QWidget(); lw.setLayout(left); lw.setFixedWidth(260); layout.addWidget(lw)
        # 右：分頁
        self.tabs = QTabWidget(); layout.addWidget(self.tabs, 1)
        self.tabs.addTab(self._build_record_tab(), "① 錄製示範")
        self.tabs.addTab(self._build_label_tab(), "② 標記物品")
        self.tabs.addTab(self._build_monitor_tab(), "③ 監控")
        self.tabs.currentChanged.connect(self.on_tab_changed)

    def _video_label(self) -> QLabel:
        lab = QLabel("（無畫面）"); lab.setAlignment(Qt.AlignCenter); lab.setMinimumSize(640, 480)
        lab.setStyleSheet("background: #202020; color: #aaa;"); return lab

    def _build_record_tab(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        self.rec_view = self._video_label(); v.addWidget(self.rec_view, 1)
        row = QHBoxLayout()
        self.btn_cam = QPushButton("開啟相機"); self.btn_cam.setStyleSheet(BIG_BTN); self.btn_cam.clicked.connect(self.toggle_camera); row.addWidget(self.btn_cam)
        self.btn_rec = QPushButton("開始錄製"); self.btn_rec.setStyleSheet(BIG_BTN); self.btn_rec.setEnabled(False); self.btn_rec.clicked.connect(self.toggle_record); row.addWidget(self.btn_rec)
        self.btn_import = QPushButton("匯入影片…"); self.btn_import.setStyleSheet(BIG_BTN); self.btn_import.clicked.connect(self.import_video); row.addWidget(self.btn_import)
        row.addStretch(1); v.addLayout(row)
        self.rec_status = QLabel("請先載入或新增專案，再開啟相機。錄製時請把整套動作完整做一遍。"); v.addWidget(self.rec_status)
        return w

    def _build_label_tab(self) -> QWidget:
        # 左：畫面＋時間軸＋這一步的按鈕；右：說明、清單、訓練進度與訊息（放下面會把畫面擠小）
        w = QWidget(); h = QHBoxLayout(w)
        v = QVBoxLayout(); h.addLayout(v, 1)
        side = QWidget(); side.setFixedWidth(400); right = QVBoxLayout(side); right.setContentsMargins(6, 0, 0, 0); h.addWidget(side)
        self.lab_view = self._video_label(); v.addWidget(self.lab_view, 1)
        self.lab_view.installEventFilter(self)          # 在畫面上拖拉框選物品
        self.timeline = TimelineBar(); self.timeline.seek.connect(lambda n: self.slider.setValue(n)); v.addWidget(self.timeline)
        self.slider = QSlider(Qt.Horizontal); self.slider.valueChanged.connect(self.on_slider); v.addWidget(self.slider)
        self.frame_label = QLabel("第 0 格 / 0"); v.addWidget(self.frame_label)
        # ---- 右欄
        self.step_header = QLabel(); self.step_header.setWordWrap(True); self.step_header.setStyleSheet("font-size: 16px; font-weight: bold; padding: 4px 0;"); right.addWidget(self.step_header)
        self.step_hint = QLabel(); self.step_hint.setWordWrap(True); self.step_hint.setStyleSheet("color: #555;"); right.addWidget(self.step_hint)
        right.addWidget(QLabel("已標的物品"))
        self.item_list = QListWidget(); right.addWidget(self.item_list, 1)
        self.train_bar = QProgressBar(); self.train_bar.setVisible(False); right.addWidget(self.train_bar)
        right.addWidget(QLabel("訊息"))
        self.train_out = QTextEdit(); self.train_out.setReadOnly(True); right.addWidget(self.train_out, 1)
        # ---- 物品（物件版，主要流程）分三步，一次只看到這一步要按的東西
        self.item_stack = QStackedWidget(); v.addWidget(self.item_stack)
        # 第 1 步：標記（每一輪：空盒一次、每樣物品框一次）
        s1 = QWidget(); l1 = QVBoxLayout(s1); l1.setContentsMargins(0, 0, 0, 0)
        r1 = QHBoxLayout()
        self.btn_ref = QPushButton("這一格是空盒"); self.btn_ref.setStyleSheet(BIG_BTN); self.btn_ref.setToolTip("盒子放好、裡面還沒放東西的那一格；每放一個新盒子都按一次"); self.btn_ref.clicked.connect(self.mark_empty_ref); r1.addWidget(self.btn_ref)
        self.btn_closed = QPushButton("這一格是蓋著的盒子"); self.btn_closed.setToolTip("蓋上蓋子後的那一格；標 2～3 格，監控時蓋著的盒子就不算「盒子在」（這一輪結束）"); self.btn_closed.clicked.connect(self.mark_closed_ref); r1.addWidget(self.btn_closed)
        r1.addSpacing(24); r1.addWidget(QLabel("框選的物品叫："))
        self.item_combo = QComboBox(); self.item_combo.setEditable(True); self.item_combo.setMinimumWidth(160); r1.addWidget(self.item_combo)
        b_item = QPushButton("加入框選的物品"); b_item.setStyleSheet(BIG_BTN); b_item.clicked.connect(self.add_item_seed); r1.addWidget(b_item)
        r1.addStretch(1)
        b_item_del = QPushButton("刪除選取的物品"); b_item_del.clicked.connect(self.delete_item); r1.addWidget(b_item_del)
        l1.addLayout(r1)
        # 看整張桌面：物品不一定在盒子裡（現場是先翻鞋舌檢查、再把鞋子放在蓋著的盒子上），框一個固定範圍，不跟著盒子
        r1b = QHBoxLayout()
        self.chk_fixed = QCheckBox("看整張桌面（固定範圍，不跟著盒子）"); self.chk_fixed.setToolTip("勾了之後：在畫面上框住整個工作桌面 → 按「設為桌面範圍」。「空盒」按鈕改成「空桌面」（桌上沒盒子、沒東西的那一格）")
        self.chk_fixed.toggled.connect(self._on_fixed_toggled); r1b.addWidget(self.chk_fixed)
        self.btn_region = QPushButton("把框選設為桌面範圍"); self.btn_region.clicked.connect(self.set_item_region); r1b.addWidget(self.btn_region)
        r1b.addSpacing(24)
        self.btn_box_label = QPushButton("選取的物品 ＝ 盒子的樣子（切換）"); self.btn_box_label.setToolTip("立著的盒子、翻面的盒子、鞋子放在盒子上…這些看到任一個就當盒子還在桌上；全部都看不到才算推走、這一輪結束")
        self.btn_box_label.clicked.connect(self.toggle_box_label); r1b.addWidget(self.btn_box_label)
        r1b.addStretch(1); l1.addLayout(r1b)
        # 手停在物件上的步驟（不用框、不用訓練）：檢查鞋舌這種做完沒痕跡的動作，用「手在鞋子上連續停留幾秒」判
        r1c = QHBoxLayout(); r1c.addWidget(QLabel("手停在物件上的步驟：名稱"))
        self.hand_name = QLineEdit(); self.hand_name.setPlaceholderText("例如 檢查鞋舌"); self.hand_name.setMaximumWidth(140); r1c.addWidget(self.hand_name)
        r1c.addWidget(QLabel("物件（英文）")); self.hand_prompt = QLineEdit("shoe"); self.hand_prompt.setMaximumWidth(100); r1c.addWidget(self.hand_prompt)
        r1c.addWidget(QLabel("連續停留")); self.hand_dwell = QDoubleSpinBox(); self.hand_dwell.setRange(0.2, 10.0); self.hand_dwell.setSingleStep(0.1); self.hand_dwell.setValue(0.7); self.hand_dwell.setSuffix(" 秒"); r1c.addWidget(self.hand_dwell)
        b_hand = QPushButton("加入"); b_hand.clicked.connect(self.add_hand_step); r1c.addWidget(b_hand)
        b_hand_del = QPushButton("刪除選取的手部步驟"); b_hand_del.clicked.connect(self.delete_hand_step); r1c.addWidget(b_hand_del)
        r1c.addStretch(1); l1.addLayout(r1c)
        self.item_stack.addWidget(s1)
        # 第 2 步：訓練（一顆按鈕，建立資料＋訓練自動接著跑）
        s2 = QWidget(); l2 = QVBoxLayout(s2); l2.setContentsMargins(0, 0, 0, 0)
        r2 = QHBoxLayout()
        self.btn_item_go = QPushButton("開始訓練（約 20～40 分鐘）"); self.btn_item_go.setStyleSheet(BIG_BTN + "QPushButton { background: #2d7; }"); self.btn_item_go.clicked.connect(self.start_item_training); r2.addWidget(self.btn_item_go)
        r2.addStretch(1); l2.addLayout(r2)
        self.item_stack.addWidget(s2)
        # 第 3 步：完成
        s3 = QWidget(); l3 = QVBoxLayout(s3); l3.setContentsMargins(0, 0, 0, 0)
        r3 = QHBoxLayout()
        self.btn_item_finish = QPushButton("產生步驟，前往監控"); self.btn_item_finish.setStyleSheet(BIG_BTN + "QPushButton { background: #6cf; }"); self.btn_item_finish.clicked.connect(self.finish_items); r3.addWidget(self.btn_item_finish)
        r3.addStretch(1); l3.addLayout(r3)
        self.item_stack.addWidget(s3)
        nav = QHBoxLayout(); nav.addStretch(1)
        self.btn_step_prev = QPushButton("◀ 上一步"); self.btn_step_prev.clicked.connect(lambda: self.goto_item_step(self.item_stack.currentIndex() - 1)); nav.addWidget(self.btn_step_prev)
        self.btn_step_next = QPushButton("下一步 ▶"); self.btn_step_next.setStyleSheet(BIG_BTN); self.btn_step_next.clicked.connect(lambda: self.goto_item_step(self.item_stack.currentIndex() + 1)); nav.addWidget(self.btn_step_next)
        v.addLayout(nav)
        self.goto_item_step(0); self._apply_fixed_ui(False)
        # ---- 進階（動作分段、盒內狀態）：預設收起
        self.chk_adv = QCheckBox("改用手勢／盒內狀態模式（進階；與物品模式二選一，產生步驟時會取代對方的步驟）"); v.addWidget(self.chk_adv)
        self.adv_box = QWidget(); av = QVBoxLayout(self.adv_box); av.setContentsMargins(0, 0, 0, 0)
        self.adv_box.setVisible(False); self.chk_adv.toggled.connect(self.adv_box.setVisible)
        outer_v = v; v = av
        row = QHBoxLayout()
        self.btn_start = QPushButton("設為起點（I）"); self.btn_start.setStyleSheet(BIG_BTN); self.btn_start.clicked.connect(self.set_start); row.addWidget(self.btn_start)
        self.btn_end = QPushButton("設為終點（O）"); self.btn_end.setStyleSheet(BIG_BTN); self.btn_end.clicked.connect(self.set_end); row.addWidget(self.btn_end)
        QShortcut(QKeySequence("I"), w, activated=self.set_start)
        QShortcut(QKeySequence("O"), w, activated=self.set_end)
        row.addWidget(QLabel("這一段是什麼動作："))
        self.action_combo = QComboBox(); self.action_combo.setEditable(True); self.action_combo.setMinimumWidth(200); row.addWidget(self.action_combo)
        b3 = QPushButton("加入這一段"); b3.setStyleSheet(BIG_BTN); b3.clicked.connect(self.add_segment); row.addWidget(b3)
        row.addStretch(1); v.addLayout(row)
        self.seg_hint = QLabel("起點：未設　終點：未設"); v.addWidget(self.seg_hint)
        tip = QLabel("提示：每個動作至少標 3 段（越多越準）。做完一輪請對鏡頭比一個固定手勢（例如 OK），"
                     "把那一段標成「結束」，系統就用它當一輪的結束；沒標「結束」則以「手離開畫面 3 秒」當結束。")
        tip.setWordWrap(True); tip.setStyleSheet("color: #555;"); v.addWidget(tip)
        # ---- 盒內狀態（看結果，不看手）
        srow = QHBoxLayout()
        srow.addWidget(QLabel("這一格的盒內狀態是："))
        self.state_combo = QComboBox(); self.state_combo.setEditable(True); self.state_combo.setMinimumWidth(180); srow.addWidget(self.state_combo)
        b_state = QPushButton("加入這一格的狀態"); b_state.setStyleSheet(BIG_BTN); b_state.clicked.connect(self.add_state_example); srow.addWidget(b_state)
        self.btn_state_steps = QPushButton("用狀態產生步驟"); self.btn_state_steps.setStyleSheet(BIG_BTN + "QPushButton { background: #6cf; }")
        self.btn_state_steps.clicked.connect(self.generate_state_steps); srow.addWidget(self.btn_state_steps)
        b_state_del = QPushButton("刪除選取的狀態"); b_state_del.clicked.connect(self.delete_state); srow.addWidget(b_state_del)
        b_lock = QPushButton("鎖定盒子區域"); b_lock.setToolTip("工位固定時：把目前這一格自動找到的盒子位置鎖定成固定區域，之後不再靠偵測器"); b_lock.clicked.connect(self.lock_state_region); srow.addWidget(b_lock)
        srow.addStretch(1); v.addLayout(srow)
        self.state_list = QListWidget(); self.state_list.setMaximumHeight(70); v.addWidget(self.state_list)
        self.state_hint = QLabel("盒內狀態：先加「未開」（蓋著的盒子：開盒前、蓋上後都各取幾格），再拉到「空盒」「已放卡片」「已放鞋子」……照順序各加幾格"
                                 "（每個狀態 5～10 格、不同輪都取一些），最後按「用狀態產生步驟」。"
                                 "挑格的原則：手已經離開盒子、東西看得到的那一格（手還在盒內的格程式本來就不會判）。"
                                 "第一個狀態是起始，不會變成步驟；盒子每一種樣子都要有狀態，不然程式只能從已知的狀態裡硬選一個。")
        self.state_hint.setWordWrap(True); self.state_hint.setStyleSheet("color: #555;"); v.addWidget(self.state_hint)
        self.seg_table = QTableWidget(0, 5); self.seg_table.setHorizontalHeaderLabels(["動作", "起點(格)", "終點(格)", "長度(秒)", "狀態"])
        self.seg_table.setMaximumHeight(180); v.addWidget(self.seg_table)
        row2 = QHBoxLayout()
        b4 = QPushButton("刪除選取的分段"); b4.clicked.connect(self.delete_segment); row2.addWidget(b4)
        self.btn_propose = QPushButton("自動找其他輪（依已標的分段）"); self.btn_propose.setStyleSheet(BIG_BTN)
        self.btn_propose.setToolTip("先標好第一輪（每個動作各一段），按這裡讓系統在影片其餘部分找出相同動作，當作建議；接受後才算分段。")
        self.btn_propose.clicked.connect(self.propose); row2.addWidget(self.btn_propose)
        self.btn_accept_all = QPushButton("接受全部建議"); self.btn_accept_all.clicked.connect(self.accept_all_proposals); self.btn_accept_all.setEnabled(False); row2.addWidget(self.btn_accept_all)
        self.btn_accept_one = QPushButton("接受選取的建議"); self.btn_accept_one.clicked.connect(self.accept_selected_proposal); self.btn_accept_one.setEnabled(False); row2.addWidget(self.btn_accept_one)
        self.btn_clear_props = QPushButton("清除建議"); self.btn_clear_props.clicked.connect(self.clear_proposals); self.btn_clear_props.setEnabled(False); row2.addWidget(self.btn_clear_props)
        row2.addStretch(1)
        self.btn_train = QPushButton("訓練並產生步驟"); self.btn_train.setStyleSheet(BIG_BTN + "QPushButton { background: #2d7; }")
        self.btn_train.clicked.connect(self.train); row2.addWidget(self.btn_train)
        v.addLayout(row2)
        v = outer_v; v.addWidget(self.adv_box)
        return w

    def _build_monitor_tab(self) -> QWidget:
        w = QWidget(); h = QHBoxLayout(w)
        left = QVBoxLayout()
        self.mon_view = self._video_label(); left.addWidget(self.mon_view, 1)
        row = QHBoxLayout()
        self.btn_mon = QPushButton("啟動監控"); self.btn_mon.setStyleSheet(BIG_BTN); self.btn_mon.clicked.connect(self.toggle_monitor); row.addWidget(self.btn_mon)
        self.btn_reset = QPushButton("重置／解除警告"); self.btn_reset.setStyleSheet(BIG_BTN + "QPushButton { background: #f96; }")
        self.btn_reset.clicked.connect(self.reset_monitor); row.addWidget(self.btn_reset)
        self.chk_demo = QCheckBox("用示範影片測試（不接相機）"); row.addWidget(self.chk_demo)
        b_shot = QPushButton("存截圖"); b_shot.setToolTip("把目前監控畫面（含框線與狀態文字）存到 cache\\screenshot_時間.jpg，回報問題用")
        b_shot.clicked.connect(self.save_screenshot); row.addWidget(b_shot)
        row.addStretch(1); left.addLayout(row)
        self.mon_status = QLabel("—"); left.addWidget(self.mon_status)
        h.addLayout(left, 1)
        right = QVBoxLayout(); right.addWidget(QLabel("事件"))
        self.event_list = QListWidget(); right.addWidget(self.event_list, 1)
        rw = QWidget(); rw.setLayout(right); rw.setFixedWidth(360); h.addWidget(rw)
        return w

    # ---- 專案

    def refresh_projects(self) -> None:
        self.project_list.clear(); self.project_list.addItems(list_projects())

    def _camera_source(self) -> str:
        return (self.cam_edit.text() or "0").strip()

    def _camera_resolution(self) -> tuple[int, int, float]:
        try:
            cam = CameraSource(self._camera_source()); r = (cam.resolution[0], cam.resolution[1], cam.fps); cam.release(); return r
        except Exception:
            return (1280, 720, 30.0)

    def new_project(self) -> None:
        name, ok = QInputDialog.getText(self, "新增專案", "專案名稱：")
        if not ok or not name.strip():
            return
        w, h, fps = self._camera_resolution()
        try:
            create_project_with_defaults(name.strip(), w, h, fps)
        except ProjectConfigError as e:
            QMessageBox.warning(self, "無法建立", str(e)); return
        self.refresh_projects(); self.load_project(name.strip())

    def load_selected(self) -> None:
        item = self.project_list.currentItem()
        if item is None:
            QMessageBox.information(self, "提示", "請先在清單選一個專案"); return
        self.load_project(item.text())

    def _reset_views(self) -> None:
        """換專案時把三個分頁的畫面、訊息、暫存狀態全部歸零（之前只重讀資料，畫面上還留著上一個專案的東西）。"""
        self.rec_view.setPixmap(QPixmap()); self.rec_view.setText("（無畫面）")
        self.rec_status.setText("請先載入或新增專案，再開啟相機。錄製時請把整套動作完整做一遍。")
        self.lab_view.setPixmap(QPixmap()); self.lab_view.setText("（無畫面）")
        self._cur_frame = None; self._cur_n = 0; self._sel_rect = None; self._drag = None
        self.seg_start = None; self.seg_end = None; self._update_hint()
        self.timeline.set_state(0, [], None, None, 0, [])
        self.train_out.clear(); self.train_bar.setVisible(False)
        self.chk_adv.setChecked(False)
        self.mon_view.setPixmap(QPixmap()); self.mon_view.setText("（無畫面）")
        self.mon_status.setText("—"); self.event_list.clear(); self._last_vis = None
        self.item_stack.setCurrentIndex(0)

    def load_project(self, name: str) -> None:
        if self._item_busy():
            QMessageBox.information(self, "提示", "物品模型正在建立資料或訓練中，等它跑完再換專案"); return
        self.stop_all()
        try:
            self.project = load_project(name)
        except ProjectConfigError as e:
            QMessageBox.critical(self, "專案設定有誤", str(e)); return
        self._reset_views()
        p = self.project
        steps = "、".join(s.step_name for s in p.steps) or "（尚未訓練）"
        self.project_label.setText(f"目前專案：{p.project_name}\n解析度 {p.resolution[0]}x{p.resolution[1]}\n步驟：{steps}")
        self.segments = training.load_segments(p)
        self.proposals = []
        self.refresh_segment_table()
        self._lib = None
        self.refresh_state_list()
        self._seeds = None; self._sel_rect = None
        self.refresh_item_list()
        self.open_reader()

    def delete_selected(self) -> None:
        item = self.project_list.currentItem()
        if item is None:
            return
        if QMessageBox.question(self, "確認刪除", f"確定刪除專案「{item.text()}」？（含影片與模型，無法復原）") != QMessageBox.Yes:
            return
        self.stop_all()
        if self.project and self.project.project_name == item.text():
            self.project = None; self.project_label.setText("（尚未載入專案）")
        delete_project(item.text()); self.refresh_projects()

    def save_project_config(self) -> None:
        if not self.project:
            return
        with open(os.path.join(self.project.dir_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(project_to_dict(self.project), f, ensure_ascii=False, indent=2)

    # ---- ① 錄製

    def toggle_camera(self) -> None:
        if self.rec_worker is not None:
            self.rec_worker.stop(); self.rec_worker.wait(3000); self.rec_worker = None
            self.btn_cam.setText("開啟相機"); self.btn_rec.setEnabled(False); self.btn_rec.setText("開始錄製")
            return
        if not self.project:
            QMessageBox.information(self, "提示", "請先載入或新增專案"); return
        self.rec_worker = RecordWorker(self._camera_source(), self.project.resolution[0], self.project.resolution[1])
        self.rec_worker.frame_ready.connect(self.on_rec_frame)
        self.rec_worker.info.connect(self.rec_status.setText)
        self.rec_worker.failed.connect(self.on_camera_failed)
        self.rec_worker.start(); self.btn_cam.setText("關閉相機"); self.btn_rec.setEnabled(True)

    @Slot(str)
    def on_camera_failed(self, msg: str) -> None:
        """相機開不起來或中途斷線。錄到一半的影片要保住，照正常停止的流程收尾。"""
        rw = self.rec_worker
        was_recording = rw is not None and (rw.recording or bool(rw.record_path))
        if rw is not None and was_recording:
            n = rw.stop_recording()
            self._finalize_recording(rw, n, note=f"相機中途斷線（{msg}），已保存錄到的 {n} 格。")
        self.toggle_camera()
        if not was_recording:
            QMessageBox.critical(self, "相機錯誤", msg)

    def _finalize_recording(self, rw: RecordWorker, n: int, note: str = "") -> None:
        """錄完：把實際解析度／fps 寫進專案、清掉舊分段、打開時間軸。"""
        self.btn_rec.setText("開始錄製")
        p = self.project
        if p is None or n <= 0:
            return
        p.video_file = "demo.avi"; p.resolution = list(rw.resolution); p.fps = float(rw.fps)
        self.segments = []; training.save_segments(p, self.segments)
        self.save_project_config(); self.refresh_segment_table(); self.open_reader()
        self.rec_status.setText(f"{note}已存 {n} 格（{n / max(p.fps, 1):.1f} 秒）到 {p.video_path}。到「② 分段說明動作」標記。")
        rw.record_path = ""
        self.tabs.setCurrentIndex(1)

    @Slot(object)
    def on_rec_frame(self, frame) -> None:
        self.rec_view.setPixmap(bgr_to_pixmap(frame, self.rec_view.width(), self.rec_view.height()))
        if self.rec_worker and self.rec_worker.recording:
            self.rec_status.setText(f"● 錄製中… 已錄 {self.rec_worker._rec.frame_count / max(self.rec_worker.fps, 1):.1f} 秒")

    def toggle_record(self) -> None:
        if not self.rec_worker or not self.project:
            return
        if self.rec_worker.recording:
            n = self.rec_worker.stop_recording()
            self._finalize_recording(self.rec_worker, n)
        else:
            self.rec_worker.start_recording(os.path.join(self.project.dir_path, "demo.avi"))
            self.btn_rec.setText("停止錄製")

    def import_video(self) -> None:
        if not self.project:
            QMessageBox.information(self, "提示", "請先載入或新增專案"); return
        path, _ = QFileDialog.getOpenFileName(self, "選擇影片", "", "影片 (*.mp4 *.avi *.mov *.mkv)")
        if not path:
            return
        try:
            name, w, h, fps = import_demo_video(path, self.project.dir_path)
        except Exception as e:
            QMessageBox.critical(self, "匯入失敗", str(e)); return
        p = self.project; p.video_file = name; p.resolution = [w, h]; p.fps = fps
        self.segments = []; training.save_segments(p, self.segments); self.save_project_config()
        self.refresh_segment_table(); self.open_reader()
        self.rec_status.setText(f"已匯入 {name}（{w}x{h} @ {fps:.1f} fps）"); self.tabs.setCurrentIndex(1)

    # ---- ② 分段

    def open_reader(self) -> None:
        if self.reader is not None:
            self.reader.release(); self.reader = None
        if not self.project or not os.path.isfile(self.project.video_path):
            self.slider.setRange(0, 0); self.frame_label.setText("（這個專案還沒有示範影片）"); return
        try:
            self.reader = FrameReader(self.project.video_path)
        except Exception as e:
            self.frame_label.setText(f"影片開不起來：{e}"); return
        self.slider.setRange(0, max(self.reader.frame_count - 1, 0)); self.slider.setValue(0); self.on_slider(0)

    @Slot(int)
    def on_slider(self, n: int) -> None:
        if self.reader is None:
            return
        try:
            frame = self.reader.get_frame(n)
        except Exception as e:
            self.frame_label.setText(str(e)); return
        self._cur_frame = frame; self._cur_n = n
        if getattr(self, "_drag", None) is None:
            self._sel_rect = None                     # 換格就取消還沒加入的框選
        self._render_label()
        self.frame_label.setText(f"第 {n} 格 / {self.reader.frame_count - 1}（{n / max(self.reader.fps, 1):.2f} 秒）")
        self._update_timeline()

    def _update_timeline(self) -> None:
        total = self.reader.frame_count if self.reader else 0
        self.timeline.set_state(total, self.segments, self.seg_start, self.seg_end, self.slider.value(), self.proposals)

    # ---- 盒內狀態

    def _state_lib(self) -> StateLibrary:
        if getattr(self, "_lib", None) is None or self._lib_dir != self.project.dir_path:
            self._lib = StateLibrary(self.project.dir_path); self._lib_dir = self.project.dir_path
        return self._lib

    def refresh_state_list(self) -> None:
        self.state_list.clear()
        if not self.project:
            return
        lib = self._state_lib()
        counts = lib.counts()
        for i, st in enumerate(lib.order):
            self.state_list.addItem(f"{i + 1}. {st}　{counts.get(st, 0)} 格" + ("（起始）" if i == 0 else "") + ("（一輪結束）" if st in training.STATE_END_LABELS else ""))
        cur = self.state_combo.currentText()
        self.state_combo.clear(); self.state_combo.addItems(lib.order)
        if cur and cur not in lib.order:
            self.state_combo.setEditText(cur)

    def _detect_box_region(self, frame):
        """在這一格找盒子並算出狀態區域；專案沒有錨點就自動加 box 錨點。"""
        p = self.project
        changed = training.ensure_state_anchor(p)
        if changed:
            self.project = load_project(p.project_name); p = self.project
        if p.state_fixed:
            return tuple(p.state_fixed)
        if getattr(self, "_detector", None) is None:
            from object_detector import ObjectDetector
            self._detector = ObjectDetector(YOLO_WEIGHTS, [a.prompt for a in p.anchors])
        elif set(self._detector.prompts) != {a.prompt for a in p.anchors}:
            from object_detector import ObjectDetector
            self._detector = ObjectDetector(YOLO_WEIGHTS, [a.prompt for a in p.anchors])
        tr = _SceneTracker(p.anchors, tuple(p.work_area) if p.work_area else None)
        tr.update_detections(self._detector.detect(frame), 0.0)
        return state_region(p, tr.anchors_at(0.0))

    def add_state_example(self) -> None:
        if not self.project or self.reader is None:
            QMessageBox.information(self, "提示", "還沒有示範影片"); return
        name = self.state_combo.currentText().strip()
        if not name:
            QMessageBox.information(self, "提示", "請輸入這一格的盒內狀態名稱（例如 空盒）"); return
        if not os.path.isfile(YOLO_WEIGHTS) and not self.project.state_fixed:
            QMessageBox.critical(self, "缺模型檔", f"找不到 {YOLO_WEIGHTS}，無法自動找盒子；可改用「鎖定盒子區域」"); return
        n = self.slider.value()
        frame = self.reader.get_frame(n)
        try:
            # 這一格偵測不到就往前最多 15 格找（監控時錨點也會沿用上一次偵測，盒子沒動位置就一樣）
            region = None
            for back in (0, 5, 10, 15):
                if n - back < 0:
                    break
                region = self._detect_box_region(frame if back == 0 else self.reader.get_frame(n - back))
                if region is not None:
                    break
        except Exception as e:
            QMessageBox.critical(self, "找盒子失敗", str(e)); return
        if region is None:
            QMessageBox.information(self, "找不到盒子", "這一格（和前 15 格）都偵測不到盒子，換一格再試；或在有盒子的那一格按「鎖定盒子區域」"); return
        x, y, w, h = [int(v) for v in region]
        crop = frame[max(0, y):y + h, max(0, x):x + w]
        if crop.size == 0:
            QMessageBox.information(self, "區域無效", "盒子區域超出畫面"); return
        self._state_lib().add_example(name, crop, n)
        self.refresh_state_list()
        # 畫面上框出剛裁的區域
        vis = frame.copy(); cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 4)
        self.lab_view.setPixmap(bgr_to_pixmap(vis, self.lab_view.width(), self.lab_view.height()))
        loo = self._state_lib().leave_one_out()
        self.train_out.setPlainText(f"已加入「{name}」第 {self._state_lib().counts()[name]} 格（第 {n} 格，區域 {x},{y},{w},{h}）。"
                                    + (f"\n範例互驗：{loo['accuracy']:.0%}（{loo['n']} 格）" + (f"，混淆：{loo['confusions']}" if loo["confusions"] else "") if loo["n"] >= 4 else ""))

    def delete_state(self) -> None:
        row = self.state_list.currentRow()
        lib = self._state_lib() if self.project else None
        if lib is None or row < 0 or row >= len(lib.order):
            return
        st = lib.order[row]
        if QMessageBox.question(self, "確認", f"刪除狀態「{st}」的全部範例？") != QMessageBox.Yes:
            return
        lib.remove_state(st); self.refresh_state_list()

    def lock_state_region(self) -> None:
        if not self.project or self.reader is None:
            return
        frame = self.reader.get_frame(self.slider.value())
        try:
            self.project.state_fixed = None
            region = self._detect_box_region(frame)
        except Exception as e:
            QMessageBox.critical(self, "找盒子失敗", str(e)); return
        if region is None:
            QMessageBox.information(self, "找不到盒子", "這一格偵測不到盒子，換一格再試"); return
        self.project.state_fixed = [int(v) for v in region]
        self.save_project_config()
        self.train_out.setPlainText(f"已鎖定盒子區域 {self.project.state_fixed}（之後不再靠偵測器；要解除請把 config.json 的 state_fixed 設為 null）")

    def generate_state_steps(self) -> None:
        if not self.project:
            return
        lib = self._state_lib()
        if len(lib.order) < 2:
            QMessageBox.information(self, "提示", "至少要有 2 個狀態（例如 空盒、已放鞋子）"); return
        few = [s for s in lib.order if lib.counts().get(s, 0) < 3]
        if few and QMessageBox.question(self, "範例偏少", f"這些狀態不到 3 格：{'、'.join(few)}。範例少容易認錯，仍要產生步驟嗎？") != QMessageBox.Yes:
            return
        names = training.apply_state_steps(self.project, lib.order)
        self.load_project(self.project.project_name)
        p = self.project
        end_txt = f"盒內變成「{p.cycle_end.label}」" if p.cycle_end.type == "box_state" else "盒子被拿走（2 秒偵測不到盒子）"
        self.train_out.setPlainText(f"已產生步驟：{'、'.join(names) or '（無）'}\n起始狀態：{lib.order[0]}；一輪結束：{end_txt}\n到「③ 監控」測試。")

    # ---- 建議分段（自動找其他輪）

    def propose(self) -> None:
        if not self.project or self.reader is None:
            QMessageBox.information(self, "提示", "還沒有示範影片"); return
        if not self.segments:
            QMessageBox.information(self, "提示", "請先把第一輪標好（每個動作各一段），系統才知道要找什麼"); return
        if not os.path.isfile(HAND_MODEL):
            QMessageBox.critical(self, "缺模型檔", f"找不到 {HAND_MODEL}"); return
        self.btn_propose.setEnabled(False); self.train_bar.setVisible(True); self.train_bar.setRange(0, 100)
        self.propose_worker = ProposeWorker(self.project.dir_path)
        self.propose_worker.progress.connect(self.on_train_progress)
        self.propose_worker.done.connect(self.on_proposals)
        self.propose_worker.failed.connect(lambda m: (self.btn_propose.setEnabled(True), self.train_bar.setVisible(False),
                                                     QMessageBox.critical(self, "自動找其他輪失敗", m)))
        self.propose_worker.start()

    @Slot(list)
    def on_proposals(self, props: list) -> None:
        self.btn_propose.setEnabled(True); self.train_bar.setVisible(False)
        self.proposals = list(props)
        self.refresh_segment_table()
        n = len(props)
        if n == 0:
            self.train_out.setPlainText("沒有找到像已標動作的段落。可能第一輪標得太短、或其他輪的動作差太多；多標一輪再試。")
        else:
            labs = {}
            for p in props:
                labs[p["label"]] = labs.get(p["label"], 0) + 1
            self.train_out.setPlainText(f"找到 {n} 段建議（{'、'.join(f'{k} {v}' for k, v in labs.items())}）。"
                                        "時間軸上淡色斜線塊、表格「狀態」欄寫「建議」的就是。請拉到每一段看一下：對的按「接受」，不對的留著不管或「清除建議」。"
                                        "接受一輪後再按一次「自動找其他輪」會更準。")

    def _proposal_at_row(self, row: int) -> int | None:
        idx = row - len(self.segments)
        return idx if 0 <= idx < len(self.proposals) else None

    def accept_selected_proposal(self) -> None:
        idx = self._proposal_at_row(self.seg_table.currentRow())
        if idx is None:
            QMessageBox.information(self, "提示", "請在表格選一列「建議」"); return
        self._accept([idx])

    def accept_all_proposals(self) -> None:
        self._accept(list(range(len(self.proposals))))

    def _accept(self, idxs: list[int]) -> None:
        accepted = 0
        for i in sorted(idxs, reverse=True):
            p = self.proposals[i]
            if training.overlapping_segment(self.segments, p["start"], p["end"]) is None:
                self.segments.append({"label": p["label"], "start": int(p["start"]), "end": int(p["end"])}); accepted += 1
            del self.proposals[i]
        self.segments.sort(key=lambda s: s["start"])
        training.save_segments(self.project, self.segments)
        self.refresh_segment_table()
        self.train_out.setPlainText(f"已接受 {accepted} 段。現在共 {len(self.segments)} 段。")

    def clear_proposals(self) -> None:
        self.proposals = []; self.refresh_segment_table()

    def set_start(self) -> None:
        if self.reader is None:
            return
        self.seg_start = self.slider.value(); self._update_hint()

    def set_end(self) -> None:
        if self.reader is None:
            return
        self.seg_end = self.slider.value(); self._update_hint()

    def _update_hint(self) -> None:
        s = "未設" if self.seg_start is None else str(self.seg_start)
        e = "未設" if self.seg_end is None else str(self.seg_end)
        on = BIG_BTN + "QPushButton { background: %s; color: white; font-weight: bold; }"
        self.btn_start.setText("設為起點（I）" if self.seg_start is None else f"起點 ✓ 第 {self.seg_start} 格")
        self.btn_start.setStyleSheet(BIG_BTN if self.seg_start is None else on % "#2a9d4a")
        self.btn_end.setText("設為終點（O）" if self.seg_end is None else f"終點 ✓ 第 {self.seg_end} 格")
        self.btn_end.setStyleSheet(BIG_BTN if self.seg_end is None else on % "#c84a4a")
        if self.seg_start is not None and self.seg_end is not None:
            a, b = sorted((self.seg_start, self.seg_end))
            fps = self.project.fps if self.project else 30.0
            self.seg_hint.setText(f"起點：{s}　終點：{e}　長度 {(b - a + 1) / max(fps, 1):.1f} 秒 → 輸入動作名稱後按「加入這一段」")
        else:
            self.seg_hint.setText(f"起點：{s}　終點：{e}")
        self._update_timeline()

    def add_segment(self) -> None:
        if self.reader is None:
            QMessageBox.information(self, "提示", "還沒有示範影片"); return
        label = self.action_combo.currentText().strip()
        if self.seg_start is None or self.seg_end is None:
            QMessageBox.information(self, "提示", "請先設起點與終點"); return
        if not label:
            QMessageBox.information(self, "提示", "請輸入這一段是什麼動作"); return
        a, b = sorted((self.seg_start, self.seg_end))
        if b - a < 30:
            QMessageBox.information(self, "太短", f"這一段只有 {b - a} 格（不到 1 秒），請拉長一點"); return
        hit = training.overlapping_segment(self.segments, a, b + 1)
        if hit is not None:
            QMessageBox.information(
                self, "跟已有的分段重疊",
                f"這一段（{a}～{b}）跟已標的「{hit['label']}」（{hit['start']}～{hit['end'] - 1}）重疊。\n"
                "同一個時間只能屬於一段；「多標幾段」是指同一個動作「做好幾次」、各標一段，不是把同一次重複框。\n"
                "影片裡沒有第二次的話，請重錄一段 60 秒左右、每個動作做 3 次以上。")
            return
        self.segments.append({"label": label, "start": a, "end": b + 1})
        self.proposals = [p for p in self.proposals if not (a < p["end"] and p["start"] < b + 1)]   # 手動標掉的範圍，建議就不用了
        training.save_segments(self.project, self.segments)
        self.seg_start = self.seg_end = None; self._update_hint(); self.refresh_segment_table()
        if self.action_combo.findText(label) < 0:
            self.action_combo.addItem(label)

    def delete_segment(self) -> None:
        row = self.seg_table.currentRow()
        if row < 0:
            return
        if row < len(self.segments):
            del self.segments[row]; training.save_segments(self.project, self.segments)
        else:
            idx = self._proposal_at_row(row)
            if idx is not None:
                del self.proposals[idx]
        self.refresh_segment_table()

    def refresh_segment_table(self) -> None:
        fps = self.project.fps if self.project else 30.0
        rows = [(s, "已標") for s in self.segments] + [(p, f"建議 {p.get('prob', 0):.0%}") for p in self.proposals]
        self.seg_table.setRowCount(len(rows))
        for i, (s, state) in enumerate(rows):
            for j, val in enumerate((s["label"], s["start"], s["end"] - 1, f"{(s['end'] - s['start']) / max(fps, 1):.1f}", state)):
                item = QTableWidgetItem(str(val))
                if state != "已標":
                    item.setForeground(QColor(120, 90, 0)); item.setBackground(QColor(255, 246, 200))
                self.seg_table.setItem(i, j, item)
        has_props = bool(self.proposals)
        for b in (getattr(self, "btn_accept_all", None), getattr(self, "btn_accept_one", None), getattr(self, "btn_clear_props", None)):
            if b is not None:
                b.setEnabled(has_props)
        labels = sorted({s["label"] for s in self.segments})
        current_text = self.action_combo.currentText()
        self.action_combo.clear(); self.action_combo.addItems(labels)
        if current_text and current_text not in labels:
            self.action_combo.setEditText(current_text)
        self._update_timeline()

    def train(self) -> None:
        if not self.project:
            return
        if not self.segments:
            QMessageBox.information(self, "提示", "還沒有分段，請先標記"); return
        if not os.path.isfile(HAND_MODEL):
            QMessageBox.critical(self, "缺模型檔", f"找不到 {HAND_MODEL}"); return
        # 訓練前先檢查每個動作的段數，不夠就講清楚差幾段；可以選擇先訓練看流程
        min_clips = 3
        counts = training.segment_counts(self.segments)
        raw = {}
        for s in self.segments:
            raw[s["label"]] = raw.get(s["label"], 0) + 1
        short = {l: counts.get(l, 0) for l in raw if counts.get(l, 0) < min_clips and l != training.OTHER_LABEL}
        if short:
            detail = "\n".join(f"　「{l}」：有效 {n} 段（標了 {raw[l]} 段，重疊的只算一次；還差 {min_clips - n} 段）" for l, n in short.items())
            ans = QMessageBox.question(
                self, "分段不夠",
                f"每個動作至少要 {min_clips} 段，驗證出來的準確率才有意義。目前：\n{detail}\n\n"
                "要先用現有的分段訓練、看看流程嗎？（這樣訓出來的模型只能看流程，準確率數字沒有意義）\n"
                "選「No」回時間軸多標幾段。",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ans != QMessageBox.Yes:
                return
            min_clips = 1
        self.btn_train.setEnabled(False); self.train_bar.setVisible(True); self.train_bar.setRange(0, 100); self.train_out.setPlainText("")
        self.train_worker = TrainWorker(self.project.dir_path, min_clips=min_clips)
        self.train_worker.progress.connect(self.on_train_progress)
        self.train_worker.done.connect(self.on_train_done)
        self.train_worker.failed.connect(self.on_train_failed)
        self.train_worker.start()

    @Slot(str, int, int)
    def on_train_progress(self, stage: str, done: int, total: int) -> None:
        self.train_bar.setValue(int(done / total * 100) if total else 0); self.train_bar.setFormat(f"{stage} %p%")

    @Slot(object)
    def on_train_done(self, r) -> None:
        self.btn_train.setEnabled(True); self.train_bar.setVisible(False)
        text = r.report
        if r.model_path:
            step_labels = [l for l in r.labels if l != training.OTHER_LABEL and l not in training.END_LABELS]
            end_labels = [l for l in r.labels if l in training.END_LABELS]
            text += f"\n\n已產生步驟：{'、'.join(step_labels)}"
            text += f"\n一輪結束手勢：{'、'.join(end_labels)}" if end_labels else "\n一輪結束：手離開畫面 3 秒（沒有標「結束」手勢）"
            text += f"\n整體正確率 {r.accuracy:.0%}；低於 80% 建議多錄幾段再訓練。"
            if r.skipped:
                text += f"\n這些動作分段太少被跳過：{'、'.join(r.skipped)}"
            self.load_project(self.project.project_name)
        self.train_out.setPlainText(text)

    @Slot(str)
    def on_train_failed(self, msg: str) -> None:
        self.btn_train.setEnabled(True); self.train_bar.setVisible(False)
        QMessageBox.critical(self, "訓練失敗", msg)

    # ---- ③ 監控

    def toggle_monitor(self) -> None:
        if self.mon_worker is not None:
            self.mon_worker.stop(); self.mon_worker.wait(5000); self.mon_worker = None
            self.btn_mon.setText("啟動監控"); self.btn_mon.setEnabled(True); return
        if not self.project:
            QMessageBox.information(self, "提示", "請先載入專案"); return
        if not self.project.steps:
            QMessageBox.information(self, "提示", "這個專案還沒有步驟，請先在「② 分段說明動作」訓練"); return
        source = self.project.video_path if self.chk_demo.isChecked() else self._camera_source()
        if self.chk_demo.isChecked() and not os.path.isfile(source):
            QMessageBox.information(self, "提示", "沒有示範影片"); return
        self.event_list.clear()
        self.mon_worker = MonitorWorker(self.project, source)
        self.mon_worker.frame_ready.connect(self.on_mon_frame)
        self.mon_worker.events.connect(self.on_mon_events)
        self.mon_worker.info.connect(self.mon_status.setText)
        self.mon_worker.failed.connect(lambda m: (QMessageBox.critical(self, "監控錯誤", m), self.toggle_monitor()))
        self.mon_worker.finished.connect(lambda: (self.btn_mon.setText("啟動監控"), self.btn_mon.setEnabled(True)))
        # 第一張畫面前按鈕先鎖住：模型載入要 10～30 秒，畫面還沒動時再按一次會把監控停掉，看起來像「沒反應」
        self.mon_status.setText("載入中（第一次約 10～30 秒，請等畫面出現）…")
        self.btn_mon.setEnabled(False)
        self.mon_worker.start(); self.btn_mon.setText("停止監控")

    @Slot(object, object)
    def on_mon_frame(self, vis, snap) -> None:
        if not self.btn_mon.isEnabled():
            self.btn_mon.setEnabled(True)
        self._last_vis = vis
        self.mon_view.setPixmap(bgr_to_pixmap(vis, self.mon_view.width(), self.mon_view.height()))
        self.mon_status.setText(f"完成 {snap['cycle_count']} 輪　漏做 {snap['bad_cycle_count']} 輪　"
                                + ("⚠ " + snap["alert_message"] if snap["alert_active"] else ("等待下一輪" if not snap["armed"] else "監控中")))

    def save_screenshot(self) -> None:
        vis = getattr(self, "_last_vis", None)
        if vis is None:
            QMessageBox.information(self, "提示", "還沒有監控畫面"); return
        out = os.path.join(app_base_dir(), "cache", f"screenshot_{datetime.now():%Y%m%d_%H%M%S}.jpg")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        write_image(out, vis)
        self.mon_status.setText(f"已存截圖：{out}")
        self.mon_status.setText(f"完成 {snap['cycle_count']} 輪　漏做 {snap['bad_cycle_count']} 輪　"
                                + ("⚠ " + snap["alert_message"] if snap["alert_active"] else ("等待下一輪" if not snap["armed"] else "監控中")))

    @Slot(list)
    def on_mon_events(self, evs: list) -> None:
        for ev in evs:
            if ev["event_type"] in ("cycle_done", "alert_missing", "alert_skip", "alert_timeout", "reset"):
                name = {"cycle_done": "完成一輪", "alert_missing": "漏做", "alert_skip": "跳步", "alert_timeout": "超時", "reset": "重置"}[ev["event_type"]]
                self.event_list.insertItem(0, f"{datetime.now():%H:%M:%S}  {name}  {ev['message']}")

    def reset_monitor(self) -> None:
        if self.mon_worker is not None:
            self.mon_worker.request_reset()

    # ---- 物品（物件版）

    def _item_seeds(self) -> items.ItemSeeds:
        if getattr(self, "_seeds", None) is None or self._seeds_dir != self.project.dir_path:
            self._seeds = items.ItemSeeds(self.project.dir_path); self._seeds_dir = self.project.dir_path
        return self._seeds

    def _render_label(self) -> None:
        """目前這一格 + 疊加：已標的物品框（綠）、空盒參考（文字）、正在框選的（黃）。"""
        frame = getattr(self, "_cur_frame", None)
        if frame is None:
            return
        vis = frame
        n = getattr(self, "_cur_n", 0)
        overlays = []
        if self.project:
            seeds = self._item_seeds()
            overlays = [(s.name + ("（自動）" if s.auto else ""), s.rect, (150, 240, 150) if s.auto else (0, 220, 0)) for s in seeds.seeds if s.frame == n]
            if n in seeds.refs:
                overlays.append(("空桌面參考" if seeds.region else "空盒參考", None, (0, 200, 255)))
            if n in seeds.closed:
                overlays.append(("蓋著的盒子", None, (0, 200, 255)))
            if seeds.region:
                overlays.append(("桌面範圍", seeds.region, (255, 160, 0)))
        if overlays or self._sel_rect:
            vis = frame.copy()
            for name, rect, col in overlays:
                if rect is None:
                    cv2.putText(vis, name, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3); continue
                x, y, w, h = rect; cv2.rectangle(vis, (x, y), (x + w, y + h), col, 4)
                cv2.putText(vis, name, (x, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 3)
            if self._sel_rect:
                x, y, w, h = self._sel_rect; cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 4)
        self.lab_view.setPixmap(bgr_to_pixmap(vis, self.lab_view.width(), self.lab_view.height()))

    def _label_to_frame(self, pos) -> tuple[int, int] | None:
        """QLabel 座標 → 影格座標（pixmap 置中、等比縮放）。"""
        pm = self.lab_view.pixmap(); frame = getattr(self, "_cur_frame", None)
        if pm is None or pm.isNull() or frame is None:
            return None
        ox = (self.lab_view.width() - pm.width()) / 2; oy = (self.lab_view.height() - pm.height()) / 2
        sx = frame.shape[1] / pm.width(); sy = frame.shape[0] / pm.height()
        x = int((pos.x() - ox) * sx); y = int((pos.y() - oy) * sy)
        return max(0, min(frame.shape[1] - 1, x)), max(0, min(frame.shape[0] - 1, y))

    def eventFilter(self, obj, event):
        if obj is getattr(self, "lab_view", None) and getattr(self, "_cur_frame", None) is not None:
            t = event.type()
            if t == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                p = self._label_to_frame(event.position())
                if p:
                    self._drag = p; self._sel_rect = None
                return True
            if t == QEvent.MouseMove and getattr(self, "_drag", None) is not None:
                p = self._label_to_frame(event.position())
                if p:
                    x0, y0 = self._drag; x1, y1 = p
                    self._sel_rect = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)); self._render_label()
                return True
            if t == QEvent.MouseButtonRelease and getattr(self, "_drag", None) is not None:
                self._drag = None
                if self._sel_rect and (self._sel_rect[2] < 12 or self._sel_rect[3] < 12):
                    self._sel_rect = None                 # 點一下不算框選
                self._render_label()
                return True
        return super().eventFilter(obj, event)

    def refresh_item_list(self) -> None:
        self.item_list.clear()
        if not self.project:
            self.goto_item_step(self.item_stack.currentIndex()); return
        seeds = self._item_seeds(); counts = seeds.counts()
        for i, name in enumerate(seeds.order):
            self.item_list.addItem(f"{i + 1}. {name}　已框 {counts.get(name, 0)} 次" + ("　［盒子的樣子］" if name in seeds.box_labels else ""))
        for h in seeds.hand_steps:
            self.item_list.addItem(f"手停在「{h['prompt']}」上 {h['dwell']:.1f} 秒 → {h['name']}")
        ref_name = "空桌面參考" if seeds.region else "空盒參考"
        self.item_list.addItem(f"{ref_name} {len(seeds.refs)} 格　蓋著的盒子 {len(seeds.closed)} 格" + ("　模型：已訓練" if os.path.isfile(items.model_path(self.project)) else "　模型：尚未訓練")
                               + (f"　桌面範圍 {seeds.region[2]}×{seeds.region[3]}" if seeds.region else ""))
        self.chk_fixed.blockSignals(True); self.chk_fixed.setChecked(seeds.region is not None); self.chk_fixed.blockSignals(False)
        self._apply_fixed_ui(seeds.region is not None)
        cur = self.item_combo.currentText()
        self.item_combo.clear(); self.item_combo.addItems(seeds.order)
        if cur and cur not in seeds.order:
            self.item_combo.setEditText(cur)
        self.goto_item_step(self.item_stack.currentIndex())

    def goto_item_step(self, idx: int) -> None:
        """三步驟畫面：標題、上一步／下一步的可用狀態依資料狀況決定。"""
        idx = max(0, min(2, idx))
        self.item_stack.setCurrentIndex(idx)
        seeds = self._item_seeds() if self.project else None
        n_items = len(seeds.order) if seeds else 0
        has_model = bool(self.project) and os.path.isfile(items.model_path(self.project))
        titles = ["第 1 步／3　標記：每一輪按一次「空盒」、每樣物品框一次",
                  "第 2 步／3　訓練：一顆按鈕，等它跑完",
                  "第 3 步／3　完成：產生步驟，到監控試"]
        status = ["" if n_items == 0 else f"（目前 {n_items} 樣物品、{sum(seeds.counts().values())} 次框選、空盒參考 {len(seeds.refs)} 格）",
                  "" if not has_model else "（已有訓練好的模型；再框了新東西才需要重訓）",
                  "" if not self.project or not any(s.type == 'item' for s in self.project.steps) else "（步驟已產生）"]
        self.step_header.setText(titles[idx] + status[idx])
        hints = ["每一輪做兩件事：\n(1) 拉到盒子放好、還沒放東西的那一格 → 按「這一格是空盒」。\n"
                 "(2) 拉到東西放好的那一格 → 在畫面上按住左鍵拖拉框住它 → 輸入名稱 → 按「加入框選的物品」，每樣東西各一次。\n"
                 "流程最後會蓋上蓋子的話，再拉到蓋好的那一格按「這一格是蓋著的盒子」（標 2～3 格就夠，蓋著就當這一輪結束）。\n"
                 "東西不是放進盒子裡（例如先檢查鞋子、再把鞋放在盒子上）就勾「看整張桌面」，框一次桌面範圍，物品可以是任何動作後的樣子。\n"
                 "全部輪次做完再按「下一步」。綠框＝這一格已經框過的物品。",
                 "按「開始訓練」會自動接著做：整支影片找盒子和手（第一次約影片長度的 1～2 倍時間）→ 把你框的物品延伸成訓練資料 → 在這台電腦訓練小模型。\n"
                 "訓練時電腦會很忙，不要同時錄影或監控；訊息欄出現「訓練完成」再按「下一步」。",
                 "每樣物品會變成一個必做步驟（一輪內在盒裡看到過就算），盒子被拿走 2 秒算一輪結束。按下去就切到「③ 監控」。"]
        self.step_hint.setText(hints[idx])
        self.btn_step_prev.setEnabled(idx > 0)
        can_next = (idx == 0 and n_items > 0) or (idx == 1 and has_model)
        self.btn_step_next.setEnabled(can_next)
        self.btn_step_next.setToolTip("" if can_next else ("先框至少一樣物品" if idx == 0 else "先完成訓練" if idx == 1 else ""))
        if idx == 1:
            self.btn_item_go.setEnabled(n_items > 0 and not self._item_busy())

    def _item_busy(self) -> bool:
        w = getattr(self, "item_worker", None); p = getattr(self, "item_proc", None)
        return (w is not None and w.isRunning()) or (p is not None and p.state() != QProcess.NotRunning)

    def start_item_training(self) -> None:
        """第 2 步的一顆按鈕：建立資料 → 訓練，自動接著跑。"""
        self._auto_train = True
        self.build_item_dataset()

    def finish_items(self) -> None:
        self.generate_item_steps()
        if self.project and any(s.type == "item" for s in self.project.steps):
            self.tabs.setCurrentIndex(2)

    def _apply_fixed_ui(self, fixed: bool) -> None:
        """固定桌面範圍時：「空盒」變「空桌面」、蓋著的盒子用不到（盒子推走才算結束）。"""
        self.btn_ref.setText("這一格是空桌面" if fixed else "這一格是空盒")
        self.btn_closed.setVisible(not fixed); self.btn_region.setVisible(fixed); self.btn_box_label.setVisible(fixed)

    def _on_fixed_toggled(self, on: bool) -> None:
        if not self.project:
            return
        seeds = self._item_seeds()
        if not on and seeds.region:
            seeds.set_region(None)
        self._apply_fixed_ui(on); self.refresh_item_list(); self._render_label()
        if on and not seeds.region:
            self.train_out.setPlainText("在畫面上按住左鍵拖拉，框住整個工作桌面（盒子會出現的地方都要包到），再按「把框選設為桌面範圍」。")

    def set_item_region(self) -> None:
        if not self.project or self.reader is None:
            QMessageBox.information(self, "提示", "還沒有示範影片"); return
        if not self._sel_rect:
            QMessageBox.information(self, "提示", "請先在畫面上按住滑鼠左鍵拖拉，框住整個工作桌面"); return
        self._item_seeds().set_region(self._sel_rect); self._sel_rect = None
        self.refresh_item_list(); self._render_label()
        self.train_out.setPlainText("桌面範圍已設定（橘框）。接著標「空桌面」的格、框每一樣要確認的東西；盒子的樣子（立著、翻面、鞋子放在上面…）"
                                    "框完後在右邊清單選它、按「＝盒子的樣子」。")

    def toggle_box_label(self) -> None:
        if not self.project:
            return
        row = self.item_list.currentRow(); seeds = self._item_seeds()
        if row < 0 or row >= len(seeds.order):
            QMessageBox.information(self, "提示", "請先在右邊清單選一樣物品"); return
        name = seeds.order[row]; seeds.set_box_label(name, name not in seeds.box_labels)
        self.refresh_item_list(); self.item_list.setCurrentRow(row)

    def add_hand_step(self) -> None:
        if not self.project:
            return
        name = self.hand_name.text().strip(); prompt = self.hand_prompt.text().strip()
        if not name or not prompt:
            QMessageBox.information(self, "提示", "請輸入步驟名稱和物件的英文提示（例如 shoe）"); return
        self._item_seeds().add_hand_step(name, prompt, self.hand_dwell.value()); self.refresh_item_list()
        self.train_out.setPlainText(f"已加入手部步驟「{name}」：手在「{prompt}」上連續停留 {self.hand_dwell.value():.1f} 秒就算做過。不用框、不用訓練；產生步驟時會自動加物件錨點。")

    def delete_hand_step(self) -> None:
        if not self.project:
            return
        seeds = self._item_seeds(); row = self.item_list.currentRow() - len(seeds.order)
        if row < 0 or row >= len(seeds.hand_steps):
            QMessageBox.information(self, "提示", "請先在右邊清單選一個手部步驟"); return
        seeds.remove_hand_step(seeds.hand_steps[row]["name"]); self.refresh_item_list()

    def mark_empty_ref(self) -> None:
        if not self.project or self.reader is None:
            QMessageBox.information(self, "提示", "還沒有示範影片"); return
        self._item_seeds().add_ref(self.slider.value()); self.refresh_item_list(); self._render_label()
        fixed = self._item_seeds().region is not None
        self.train_out.setPlainText(f"第 {self.slider.value()} 格已標為{'空桌面' if fixed else '空盒'}參考。接著拉到放好東西的那一格，在畫面上拖拉框住物品。")

    def mark_closed_ref(self) -> None:
        if not self.project or self.reader is None:
            QMessageBox.information(self, "提示", "還沒有示範影片"); return
        self._item_seeds().add_closed(self.slider.value()); self.refresh_item_list(); self._render_label()
        self.train_out.setPlainText(f"第 {self.slider.value()} 格已標為「蓋著的盒子」。監控時盒子一蓋上就視同拿走，這一輪結算。")

    def add_item_seed(self) -> None:
        if not self.project or self.reader is None:
            QMessageBox.information(self, "提示", "還沒有示範影片"); return
        if not self._sel_rect:
            QMessageBox.information(self, "提示", "請先在畫面上按住滑鼠左鍵拖拉，框住這一格裡的物品"); return
        name = self.item_combo.currentText().strip()
        if not name:
            QMessageBox.information(self, "提示", "請輸入這個物品的名稱（例如 標籤）"); return
        n = self.slider.value()
        self.train_out.setPlainText("追蹤前後 2 秒中…"); QApplication.processEvents()
        try:
            auto = items.track_seed(self.project.video_path, n, self._sel_rect, window_sec=2.0, fps=self.project.fps)
        except Exception:
            auto = []
        seeds = self._item_seeds(); seeds.add_seed(name, n, self._sel_rect, auto)
        self._sel_rect = None; self.refresh_item_list(); self._render_label()
        if auto:
            a, b = auto[0][0] / max(self.project.fps, 1), auto[-1][0] / max(self.project.fps, 1)
            self.train_out.setPlainText(f"已加入「{name}」（第 {n} 格），並自動在 {a:.1f}～{b:.1f} 秒找到 {len(auto)} 格（淡綠框）。"
                                        "拉時間軸檢查，框錯的格請用網頁標記頁刪除，或刪掉這樣物品重框。")
        else:
            self.train_out.setPlainText(f"已加入「{name}」（第 {n} 格）。前後沒有追到相同的東西，這一輪可以多框幾格。")

    def delete_item(self) -> None:
        if not self.project:
            return
        row = self.item_list.currentRow(); seeds = self._item_seeds()
        if row < 0 or row >= len(seeds.order):
            return
        name = seeds.order[row]
        if QMessageBox.question(self, "確認", f"刪除物品「{name}」的全部框選？") != QMessageBox.Yes:
            return
        seeds.remove_item(name); self.refresh_item_list(); self._render_label()

    def build_item_dataset(self) -> None:
        if not self.project:
            return
        seeds = self._item_seeds()
        if not seeds.seeds:
            QMessageBox.information(self, "提示", "還沒有框選任何物品"); return
        if not os.path.isfile(YOLO_WEIGHTS):
            QMessageBox.critical(self, "缺模型檔", f"找不到 {YOLO_WEIGHTS}，無法找盒子"); return
        changed = training.sync_item_region(self.project) | training.ensure_state_anchor(self.project)
        if changed:
            self.project = load_project(self.project.project_name)
        self.btn_item_go.setEnabled(False); self.train_bar.setVisible(True); self.train_bar.setRange(0, 100); self.train_bar.setValue(0)
        self.train_out.setPlainText("建立訓練資料：第一次要先整支影片找盒子和手（約影片長度的 1～2 倍時間）…")
        self.item_worker = ItemDataWorker(self.project, seeds)
        self.item_worker.progress.connect(lambda stage, a, b: (self.train_bar.setValue(int(100 * a / max(1, b))), self.train_out.setPlainText(stage)))
        self.item_worker.done.connect(self._on_item_dataset_done)
        self.item_worker.failed.connect(lambda m: (self.btn_item_go.setEnabled(True), self.train_bar.setVisible(False), QMessageBox.critical(self, "建立訓練資料失敗", m)))
        self.item_worker.start()

    @Slot(object)
    def _on_item_dataset_done(self, stats: dict) -> None:
        self.btn_item_go.setEnabled(True); self.train_bar.setVisible(False)
        per = "、".join(f"{k} {v} 格" for k, v in stats["per_item"].items()) or "（無）"
        missing = [n for n in stats["names"] if stats["per_item"].get(n, 0) < 10]
        self.train_out.setPlainText(f"訓練資料：{stats['train']} 張（驗證另留 {stats['val']} 張）；每樣物品：{per}。"
                                    + (f"\n⚠ 這些物品不到 10 格：{'、'.join(missing)}——建議回第 1 步多框幾輪。" if missing else ""))
        if getattr(self, "_auto_train", False):
            self._auto_train = False
            if stats["train"] == 0:
                QMessageBox.information(self, "沒有訓練資料", "框選的物品延伸不出任何一格；請確認「這一格是空盒」有標、框選那一格盒子看得到。"); return
            self.train_item_model()

    def train_item_model(self) -> None:
        if not self.project:
            return
        data = os.path.join(items.dataset_dir(self.project), "data.yaml")
        if not os.path.isfile(data):
            QMessageBox.information(self, "提示", "請先按「① 建立訓練資料」"); return
        if getattr(self, "item_proc", None) is not None and self.item_proc.state() != QProcess.NotRunning:
            QMessageBox.information(self, "提示", "模型正在訓練中"); return
        cmd = items.train_command(self.project)
        self.item_proc = QProcess(self); self.item_proc.setProcessChannelMode(QProcess.MergedChannels)
        self.item_proc.setWorkingDirectory(app_base_dir())
        # 一定要從系統環境複製再加：QProcess.processEnvironment() 預設是空的，子程序會沒有 USERPROFILE／PATH，
        # torch／ultralytics 會死在 "Could not determine home directory"（使用者實際踩到）
        env = QProcessEnvironment.systemEnvironment(); env.insert("PYTHONUTF8", "1"); env.insert("PYTHONIOENCODING", "utf-8"); self.item_proc.setProcessEnvironment(env)
        self._item_log = []
        self.item_proc.readyReadStandardOutput.connect(self._on_item_train_output)
        self.item_proc.finished.connect(self._on_item_train_finished)
        self.btn_item_go.setEnabled(False); self.train_bar.setVisible(True); self.train_bar.setRange(0, 0)
        self.train_out.setPlainText("訓練物品模型中（第一個 epoch 要 1～2 分鐘）…")
        self.item_proc.start(cmd[0], cmd[1:])

    def _on_item_train_output(self) -> None:
        txt = bytes(self.item_proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in txt.splitlines():
            if "epoch" in line or "訓練完成" in line or "Error" in line:
                self._item_log.append(line.strip())
        self.train_out.setPlainText("\n".join(self._item_log[-8:]) or "訓練物品模型中…")

    def _on_item_train_finished(self, code: int, status) -> None:
        self.btn_item_go.setEnabled(True); self.train_bar.setVisible(False)
        mp = items.model_path(self.project) if self.project else ""
        if code == 0 and os.path.isfile(mp):
            self.train_out.setPlainText("\n".join(self._item_log[-3:]) + "\n訓練完成。按右下角「下一步」。")
        else:
            self.train_out.setPlainText("訓練失敗（結束碼 %d）：\n" % code + "\n".join(self._item_log[-6:]))
        self.refresh_item_list()

    def generate_item_steps(self) -> None:
        if not self.project:
            return
        seeds = self._item_seeds()
        if not seeds.order and not seeds.hand_steps:
            QMessageBox.information(self, "提示", "還沒有物品"); return
        if seeds.order and not os.path.isfile(items.model_path(self.project)):
            QMessageBox.information(self, "提示", "還沒有訓練好的物品模型，請先回第 2 步訓練"); return
        names = training.apply_item_steps(self.project, seeds.order)
        self.load_project(self.project.project_name)
        end = "盒子推走（盒子和「盒子的樣子」都看不到 1.5 秒）" if seeds.region else "盒子被拿走（偵測不到盒子 3.5 秒）"
        self.train_out.setPlainText(f"已產生步驟：{'、'.join(names)}\n一輪結束：{end}。到「③ 監控」測試。")
        self.goto_item_step(2)

    # ---- 共用

    def on_tab_changed(self, idx: int) -> None:
        if idx == 1 and self.reader is not None:
            self.on_slider(self.slider.value())

    def stop_all(self) -> None:
        if self.rec_worker is not None:
            self.toggle_camera()
        if self.mon_worker is not None:
            self.toggle_monitor()
        if self.reader is not None:
            self.reader.release(); self.reader = None

    def closeEvent(self, event) -> None:
        self.stop_all()
        if self.train_worker is not None and self.train_worker.isRunning():
            self.train_worker.wait(10000)
        super().closeEvent(event)


def make_app(argv=None) -> QApplication:
    from PySide6.QtGui import QFont
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft JhengHei", 11))
    return app


def main() -> int:
    app = make_app()
    win = MainWindow(); win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
