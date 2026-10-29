"""自動走查：用「假相機」（播一支影片）把操作介面的 ①錄製 → ②分段 → 訓練 → ③監控 整條走一遍，每步截圖、收事件。

usage: python tools/e2e_walkthrough.py --video <示範影片> --labels <逐格標籤 csv> [--name 自動走查] [--monitor-seconds 90]
       [--platform offscreen|windows] [--out cache/e2e]
- 假相機以半解析度、不等待（比真實時間快）把影片餵給 RecordWorker，所以 demo.avi 的格數 = 影片格數，標籤的格號可直接沿用。
- labels csv 的「等待」會標成「結束」（手閒著＝一輪結束的手勢），其餘依原名；只取連續 ≥ 45 格的段落。
- 監控用「用示範影片測試」勾選（VideoFileSource），跑 --monitor-seconds 秒後停止。
結束後印摘要並寫 <out>/report.txt。這驗的是介面與流程，不是真人手勢的準確率。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--name", default="自動走查")
ap.add_argument("--monitor-seconds", type=float, default=90.0)
ap.add_argument("--platform", default="offscreen")
ap.add_argument("--out", default=os.path.join("cache", "e2e"))
ap.add_argument("--scale", type=float, default=0.5)
ap.add_argument("--monitor-only", action="store_true", help="專案已錄過、訓練過：直接跳到③監控")
ap.add_argument("--reset-after", type=float, default=3.0, help="監控時出現警告幾秒後自動按「重置／解除警告」（模擬操作者）；0 = 不按")
args = ap.parse_args()
os.environ["QT_QPA_PLATFORM"] = args.platform
os.makedirs(args.out, exist_ok=True)

try:
    import torch  # noqa: E402,F401  torch 要在 Qt 之前載入（WinError 1114）
except Exception:
    pass
import cv2  # noqa: E402
import main_ui  # noqa: E402  （main_ui 內部也會先 torch 後 Qt）
from PySide6.QtCore import QTimer  # noqa: E402
import project_store  # noqa: E402
import training  # noqa: E402
from video_recorder import VideoFileSource  # noqa: E402

LOG: list[str] = []


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True); LOG.append(line)


STREAM = {"on": False}   # 按下「開始錄製」前假相機停在第一格不動，錄影檔的第 0 格才會等於影片的第 0 格


class FakeCamera:
    """介面以為它是相機：讀影片、縮小、不等待。介面與 CameraSource 相同。"""
    def __init__(self, cam_id=0, width=1280, height=720, measure_frames=20):
        self._src = VideoFileSource(args.video)
        w, h = self._src.resolution
        self.resolution = (int(w * args.scale) // 2 * 2, int(h * args.scale) // 2 * 2)
        self.fps = self._src.fps
        self.frame_index = -1
        self._t0 = time.perf_counter()
        self._first = None

    def read(self):
        if not STREAM["on"]:
            if self._first is None:
                fr = self._src.read()
                self._first = cv2.resize(fr, self.resolution, interpolation=cv2.INTER_AREA) if fr is not None else None
                self._pending_first = True
            time.sleep(1 / 30)
            return None if self._first is None else self._first.copy()
        if getattr(self, "_pending_first", False):
            self._pending_first = False
            self.frame_index += 1
            return self._first.copy()
        fr = self._src.read()
        if fr is None:
            return None
        self.frame_index += 1
        return cv2.resize(fr, self.resolution, interpolation=cv2.INTER_AREA)

    @property
    def time_sec(self):
        return self.frame_index / self.fps

    def release(self):
        self._src.release()


main_ui.CameraSource = FakeCamera
# 對話框全部改成記錄，不阻塞
dialogs: list[str] = []
for _n in ("information", "warning", "critical"):
    setattr(main_ui.QMessageBox, _n, staticmethod(lambda *a, **k: dialogs.append(f"{a[1]}: {a[2]}") if len(a) > 2 else None))
main_ui.QMessageBox.question = staticmethod(lambda *a, **k: main_ui.QMessageBox.Yes)

app = main_ui.make_app([])
d = os.path.join(project_store.projects_dir(), args.name)
if args.monitor_only:
    if not os.path.isdir(d):
        print(f"專案不存在：{d}"); sys.exit(1)
    win = main_ui.MainWindow(); win.resize(1360, 860); win.show()
    win.load_project(args.name)
    log(f"載入既有專案 {args.name}，直接跳到③監控")
else:
    if os.path.isdir(d):
        project_store.delete_project(args.name)
    probe = FakeCamera(); W, H, FPS = probe.resolution[0], probe.resolution[1], probe.fps; probe.release()
    main_ui.create_project_with_defaults(args.name, W, H, FPS)
    win = main_ui.MainWindow(); win.resize(1360, 860); win.show()
    win.load_project(args.name)
    log(f"建立專案 {args.name}（{W}x{H} @ {FPS:.1f} fps）")

labels = [r["label"] for r in csv.DictReader(open(args.labels, encoding="utf-8-sig"))]
RENAME = {"等待": "結束"}; SKIP = {"左鞋", "右鞋"}
segments = []; start = 0
for i in range(1, len(labels) + 1):
    if i == len(labels) or labels[i] != labels[start]:
        if i - start >= 45 and labels[start] not in SKIP:
            segments.append({"label": RENAME.get(labels[start], labels[start]), "start": start, "end": i})
        start = i

monitor_events: list[dict] = []
state = {"step": 5 if args.monitor_only else 0, "t_rec": 0.0, "t_mon": 0.0, "resets": 0}


def on_events(evs: list) -> None:
    monitor_events.extend(evs)
    if args.reset_after > 0 and any(ev["event_type"].startswith("alert_") for ev in evs):
        def do_reset():
            if win.mon_worker is not None:
                win.reset_monitor(); state["resets"] += 1
                log(f"③ 模擬操作者按「重置／解除警告」（第 {state['resets']} 次）")
        QTimer.singleShot(int(args.reset_after * 1000), do_reset)


def shot(name: str) -> None:
    app.processEvents()
    p = os.path.join(args.out, f"{name}.png")
    win.grab().save(p); log(f"截圖 {p}")


def finish() -> None:
    p = win.project
    summary = []
    if p:
        summary.append(f"步驟：{[s.step_name for s in p.steps]}；一輪結束：{p.cycle_end.type if p.cycle_end else None}"
                       + (f"（{p.cycle_end.label}）" if p.cycle_end and p.cycle_end.label else ""))
    kinds = {}
    for ev in monitor_events:
        kinds[ev["event_type"]] = kinds.get(ev["event_type"], 0) + 1
    summary.append(f"監控 {args.monitor_seconds:.0f} 秒事件統計：{kinds}；模擬按解除 {state['resets']} 次")
    for ev in monitor_events:
        if ev["event_type"] in ("cycle_done", "alert_missing", "alert_timeout", "cycle_discarded", "step_done"):
            summary.append(f"  [t={ev['t']:.1f}s] {ev['event_type']} {ev['step_name'] or ''} {ev['message']}".rstrip())
    if dialogs:
        summary.append("對話框：" + " | ".join(dialogs))
    for s in summary:
        log(s)
    with open(os.path.join(args.out, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")
    win.stop_all(); win.close(); app.quit()


def tick() -> None:
    s = state["step"]
    try:
        if s == 0:
            win.tabs.setCurrentIndex(0); win.toggle_camera(); log("① 開啟（假）相機"); state["step"] = 1
            QTimer.singleShot(1500, tick); return
        if s == 1:
            shot("1_camera"); win.toggle_record(); STREAM["on"] = True; log("① 開始錄製"); state["t_rec"] = time.perf_counter(); state["step"] = 2
            QTimer.singleShot(500, tick); return
        if s == 2:   # 等假相機把影片播完：RecordWorker 會回報「相機斷線」，介面自己收尾（保住錄影檔）
            rw = win.rec_worker
            if rw is not None and (time.perf_counter() - state["t_rec"]) < 600:
                QTimer.singleShot(500, tick); return
            p = win.project
            if not os.path.isfile(p.video_path) or win.reader is None:
                log(f"① 錄影檔沒保存或時間軸沒打開：{p.video_path}；對話框={dialogs}"); finish(); return
            log(f"① 錄製結束（相機播完被當成斷線，介面已收尾）：{win.reader.frame_count} 格，"
                f"{p.video_path}（{os.path.getsize(p.video_path) / 1048576:.0f} MB）；狀態列：{win.rec_status.text()}")
            state["step"] = 3; QTimer.singleShot(300, tick); return
        if s == 3:
            win.tabs.setCurrentIndex(1)
            n_frames = win.reader.frame_count if win.reader else 0
            win.segments = [sg for sg in segments if sg["end"] <= n_frames]
            training.save_segments(win.project, win.segments); win.refresh_segment_table()
            win.slider.setValue(min(1440, max(n_frames - 1, 0)))
            log(f"② 分段 {len(win.segments)} 段：" + "、".join(f"{sg['label']}({sg['start']}-{sg['end']})" for sg in win.segments[:8]) + " …")
            shot("2_segments"); win.train(); log("② 按下「訓練並產生步驟」"); state["step"] = 4
            QTimer.singleShot(1000, tick); return
        if s == 4:
            tw = win.train_worker
            if tw is not None and tw.isRunning():
                QTimer.singleShot(1000, tick); return
            log("② 訓練結果：\n" + win.train_out.toPlainText())
            shot("2_trained"); state["step"] = 5; QTimer.singleShot(300, tick); return
        if s == 5:
            win.tabs.setCurrentIndex(2); win.chk_demo.setChecked(True); win.toggle_monitor()
            if win.mon_worker is None:
                log("③ 監控沒啟動（可能沒有步驟）"); finish(); return
            win.mon_worker.events.connect(on_events)
            log("③ 啟動監控（示範影片來源）"); state["t_mon"] = time.perf_counter(); state["step"] = 6
            QTimer.singleShot(1000, tick); return
        if s == 6:
            el = time.perf_counter() - state["t_mon"]
            if abs(el - 30) < 0.6:
                shot("3_monitor_30s")
            if el >= args.monitor_seconds or win.mon_worker is None or not win.mon_worker.isRunning():
                shot("3_monitor_end")
                if win.mon_worker is not None:
                    win.toggle_monitor()
                log(f"③ 監控停止（{el:.0f} 秒）"); finish(); return
            QTimer.singleShot(500, tick); return
    except Exception as e:
        log(f"走查失敗：{type(e).__name__}: {e}"); finish()


QTimer.singleShot(500, tick)
app.exec()
