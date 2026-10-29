"""Set up the inner-box eval project for the box-STATE flow: box anchor (bottommost), state examples from known frames,
generate box_state steps. Then core_detector.py can monitor it."""
import os, sys, json
import numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import training
from project_store import load_project, project_to_dict
from scene import AnchorSpec, SceneTracker
from object_detector import ObjectDetector
from core_detector import state_region
from state_recognizer import StateLibrary
from video_recorder import FrameReader
D = os.path.join(ROOT, "cache", "_eval_innerbox")
# config: box anchor bottommost (內盒影片盒蓋在盒身上方)
cfg = {"schema_version": 2, "project_name": "_eval_innerbox", "video_file": "demo.mp4", "resolution": [1080, 1920], "fps": 30.0,
       "work_area": [0, 0, 820, 1920], "detect_every_n_frames": 15, "gesture_model": None, "gesture_anchors": [],
       "anchors": [{"name": "box", "prompt": "cardboard box", "min_conf": 0.2, "pick": "bottommost", "max_count": 2, "min_area": 60000, "max_age_sec": 3.0}],
       "zones": [], "state_anchor": "box", "state_rel": [0.05, 0.05, 0.9, 0.9], "state_every_n_frames": 5, "steps": []}
json.dump(cfg, open(os.path.join(D, "config.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
import shutil
shutil.rmtree(os.path.join(D, "states"), ignore_errors=True)
p = load_project(D)
SEGS = [("拿鞋盒",52,182),("開鞋盒",204,290),("放防黴片",325,522),("放鞋子",542,692),("拿鞋盒",1171,1249),("開鞋盒",1336,1466),
        ("放防黴片",1490,1651),("放鞋子",1671,1904),("拿鞋盒",2357,2528),("開鞋盒",2548,2662),("放防黴片",2687,2868),("放鞋子",2893,3085),
        ("拿鞋盒",3430,3616),("開鞋盒",3646,3724),("放防黴片",3739,3931),("放鞋子",3950,4106),("拿鞋盒",4461,4617),("開鞋盒",4652,4761),
        ("放防黴片",4796,4905),("放鞋子",4920,5101),("拿鞋盒",5477,5689),("開鞋盒",5709,5782),("放防黴片",5812,5937),("放鞋子",5962,6154)]
# 第 1～3 輪、照 contact sheet 用眼睛挑的格（手已離開、東西看得到；蓋著的盒子開盒前與蓋上後都取），第 4～6 輪留給監控驗證
examples = [("未開", 185), ("未開", 200), ("未開", 745), ("未開", 755),
            ("空盒", 290), ("空盒", 295), ("空盒", 320),
            ("已放防黴片", 505), ("已放防黴片", 525), ("已放防黴片", 540),
            ("已放鞋子", 650), ("已放鞋子", 665), ("已放鞋子", 710),
            ("未開", 1330), ("未開", 1895), ("未開", 1905),
            ("空盒", 1470), ("空盒", 1475),
            ("已放防黴片", 1655), ("已放防黴片", 1665),
            ("已放鞋子", 1830), ("已放鞋子", 1845), ("已放鞋子", 1770),
            ("未開", 2535), ("未開", 2545),
            ("空盒", 2665), ("空盒", 2680),
            ("已放防黴片", 2850), ("已放防黴片", 2865),
            ("已放鞋子", 3005), ("已放鞋子", 3015), ("已放鞋子", 3035)]
det = ObjectDetector(os.path.join(ROOT, "models", "yolov8s-worldv2.pt"), ["cardboard box"], conf=0.2)
reader = FrameReader(p.video_path)
lib = StateLibrary(D)
miss = 0
for st, f in examples:
    fr = reader.get_frame(f)
    # 跟介面一樣：這一格偵測不到就往前最多 15 格找（監控時錨點也會沿用上一次偵測）
    region = None
    for back in (0, 5, 10, 15):
        tr = SceneTracker(p.anchors, tuple(p.work_area)); tr.update_detections(det.detect(reader.get_frame(f - back)), 0.0)
        region = state_region(p, tr.anchors_at(0.0))
        if region is not None:
            break
    if region is None:
        miss += 1; continue
    x, y, w, h = [int(v) for v in region]
    lib.add_example(st, fr[y:y + h, x:x + w], f)
reader.release()
print("範例：", lib.counts(), "；找不到盒子的格數", miss)
print("範例互驗：", lib.leave_one_out())
names = training.apply_state_steps(p, lib.order)
p2 = load_project(D)
print("步驟：", names, "；一輪結束：", p2.cycle_end.type, p2.cycle_end.label)
