"""Reproduce '啟動監控沒有反應' headlessly: load the project in the real MainWindow, tick 用示範影片, press 啟動監控,
pump events for up to 60 s and report first-frame latency / frames / dialogs / errors."""
import os, sys, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import torch  # noqa
import main_ui
from PySide6.QtWidgets import QMessageBox
msgs = []
for name in ("information", "critical", "question", "warning"):
    setattr(QMessageBox, name, staticmethod(lambda *a, _n=name, **k: (msgs.append((_n,) + tuple(str(x) for x in a[1:3])), QMessageBox.Yes)[1]))
app = main_ui.make_app()
win = main_ui.MainWindow(); win.resize(1360, 860); win.show()
win.load_project(sys.argv[1] if len(sys.argv) > 1 else "桌面狀態測試")
p = win.project
print("project:", p.project_name, "steps:", [s.step_name for s in p.steps], "cycle_end:", p.cycle_end.type, "video:", os.path.isfile(p.video_path))
win.tabs.setCurrentIndex(2); win.chk_demo.setChecked(True)
t0 = time.perf_counter(); win.toggle_monitor()
print("after click: button =", win.btn_mon.text(), "worker =", win.mon_worker is not None, "msgs =", msgs)
stats = {"frames": 0, "first": None, "status": [], "enabled_at_first": None}
print("button enabled right after click:", win.btn_mon.isEnabled(), "| status:", win.mon_status.text())
if win.mon_worker is not None:
    def on_frame(vis, snap):
        stats["frames"] += 1
        if stats["first"] is None:
            stats["first"] = time.perf_counter() - t0
            stats["enabled_at_first"] = win.btn_mon.isEnabled()
    win.mon_worker.frame_ready.connect(on_frame)
    win.mon_worker.info.connect(lambda m: stats["status"].append((round(time.perf_counter() - t0, 1), m)))
limit = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
while time.perf_counter() - t0 < limit:
    app.processEvents(); time.sleep(0.02)
    if win.mon_worker is None or (stats["frames"] >= 150):
        break
print(f"elapsed {time.perf_counter() - t0:.1f}s  first frame after {stats['first']}  frames {stats['frames']}  button '{win.btn_mon.text()}' enabled={win.btn_mon.isEnabled()}  status '{win.mon_status.text()}'  events {win.event_list.count()}  msgs {msgs}")
print("loading messages:", stats["status"], "| button enabled before first frame:", stats["enabled_at_first"])
try:
    import psutil; print("rss MB", psutil.Process().memory_info().rss // 1048576)
except Exception:
    pass
if win.mon_worker is not None:
    win.toggle_monitor()
app.processEvents()
