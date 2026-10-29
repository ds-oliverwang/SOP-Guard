"""Add state examples to the project from named runs (R<round>-<run> → state), then generate box_state steps.
Usage: desk_label.py <project_dir> <mapping.json>   mapping = {"R1-1": "空盒", ...}; '-' or missing = skip."""
import os, sys, json, shutil
import numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import torch  # noqa
import training
from project_store import load_project
from scene import Detection, SceneTracker, rect_overlap_ratio
from core_detector import state_region, hands_in_inner, crop_roi, STATE_FRESH_SEC
from state_recognizer import StateLibrary
from video_recorder import FrameReader
D = sys.argv[1]; mapping = json.load(open(sys.argv[2], encoding="utf-8"))
p = load_project(D)
WORK_AREA = mapping.pop("_work_area", None)      # 桌面左側的雜物／左上角的黑色物會被認成盒子 → 只認這個範圍內的盒子
if WORK_AREA:
    p.work_area = [int(v) for v in WORK_AREA]
tr = json.load(open(os.path.join(D, "cache_trace.json"), encoding="utf-8"))
rows = {r["f"]: r for r in tr["rows"]}; N = tr["frames"]; FPS = tr["fps"]; STEP = 5
runs = json.load(open(os.path.join(ROOT, "cache", "desk_runs", "runs.json")))["rounds"]
# replay regions exactly like desk_runs.py
sc = SceneTracker(p.anchors, tuple(p.work_area) if p.work_area else None)
regions = {}; region = None
for f in range(N):
    r = rows.get(f)
    if r and "dets" in r:
        sc.update_detections([Detection(d[0], d[1], tuple(d[2:6])) for d in r["dets"]], f / FPS)
    new = state_region(p, sc.anchors_at(f / FPS))
    if new is None:
        region = None
    elif region is None or min(rect_overlap_ratio(new, region), rect_overlap_ratio(region, new)) < 0.6:
        region = new
    regions[f] = region
shutil.rmtree(os.path.join(D, "states"), ignore_errors=True)
lib = StateLibrary(D); reader = FrameReader(p.video_path)
def parse(key, val):
    """'state' → 用該 run 的 1/3、2/3 處兩格；'state@f1,f2' → 指定格。"""
    if "@" in val:
        st, fr = val.split("@", 1)
        return st, [int(v) for v in fr.split(",")]
    ri, k = [int(v) for v in key[1:].split("-")]
    f0, f1 = runs[ri - 1][k - 1]
    return val, sorted({f0 + (f1 - f0) // 3 // STEP * STEP, f0 + 2 * (f1 - f0) // 3 // STEP * STEP})
items = [parse(k, v) for k, v in mapping.items() if v and v != "-"]
order = []
for st, _ in items:                        # keep the order of first appearance
    if st not in order:
        order.append(st)
for st in order:                           # add in state order so lib.order == flow order
    for s, picks in items:
        if s != st:
            continue
        for f in picks:
            reg = regions.get(f)
            if reg is None:
                continue
            lib.add_example(st, crop_roi(reader.get_frame(f), reg), f)
reader.release()
print("範例：", lib.counts(), "順序：", lib.order)
print("範例互驗：", lib.leave_one_out())
names = training.apply_state_steps(p, lib.order)
p2 = load_project(D)
print("步驟：", names, "；一輪結束：", p2.cycle_end.type, p2.cycle_end.object or p2.cycle_end.label)
