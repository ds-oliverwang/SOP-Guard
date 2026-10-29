"""Record YOLO (every 15 frames) + hands (every 5 frames) for a project's demo video → <project>/cache_trace.json.
Usage: trace_project.py <project_dir>"""
import os, sys, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import torch  # noqa
import training
from project_store import load_project
from object_detector import ObjectDetector
from core_detector import HandTracker
from video_recorder import VideoFileSource
D = sys.argv[1]
p = load_project(D)
if training.ensure_state_anchor(p):
    p = load_project(D); print("added box anchor")
det = ObjectDetector(os.path.join(ROOT, "models", "yolov8s-worldv2.pt"), [a.prompt for a in p.anchors], conf=0.2, threads=2)
hands = HandTracker(os.path.join(ROOT, "models", "hand_landmarker.task"))
src = VideoFileSource(p.video_path)
rows = []; t0 = time.perf_counter(); n = 0
while True:
    fr = src.read()
    if fr is None:
        break
    t = src.time_sec; row = {"f": n, "t": round(t, 3)}
    if n % 15 == 0:
        row["dets"] = [[d.label, round(d.conf, 3)] + [round(v, 1) for v in d.xyxy] for d in det.detect(fr)]
    if n % 5 == 0:
        row["hands"] = [[round(float(x), 1), round(float(y), 1)] for x, y in hands.detect(fr, int(t * 1000))]
    if "dets" in row or "hands" in row:
        rows.append(row)
    n += 1
    if n % 1500 == 0:
        print(f"{n} frames, {time.perf_counter() - t0:.0f}s", flush=True)
hands.close(); src.release()
out = os.path.join(D, "cache_trace.json")
json.dump({"fps": p.fps, "frames": n, "rows": rows}, open(out, "w", encoding="utf-8"), ensure_ascii=False)
print("done", n, "frames", f"{time.perf_counter() - t0:.0f}s ->", out)
