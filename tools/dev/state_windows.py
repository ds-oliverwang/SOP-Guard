"""Look at what the box actually shows around each state transition: hands-inside timeline per 5 frames (from the trace)
and contact sheets of the box-region crops for round 1 / round 2 transitions."""
import os, sys, json
import numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
from project_store import load_project
from scene import Detection, SceneTracker, sub_rect, rect_overlap_ratio
from core_detector import any_point_in_rect, crop_roi
from video_recorder import FrameReader
D = os.path.join(ROOT, "cache", "_eval_innerbox"); OUT = os.path.join(ROOT, "cache", "state_windows")
os.makedirs(OUT, exist_ok=True)
p = load_project(D)
trace = json.load(open(os.path.join(ROOT, "cache", "state_trace.json"), encoding="utf-8"))
rows = {r["f"]: r for r in trace["rows"]}; N = trace["frames"]; FPS = trace["fps"]; STEP = 5
tr = SceneTracker(p.anchors, tuple(p.work_area) if p.work_area else None)
anch = {}; hands = {}; last = []
for f in range(N):
    r = rows.get(f)
    if r and "dets" in r:
        tr.update_detections([Detection(d[0], d[1], tuple(d[2:6])) for d in r["dets"]], f / FPS)
    if r and "hands" in r:
        last = [tuple(h) for h in r["hands"]]
    if f % STEP == 0:
        anch[f] = tr.anchors_at(f / FPS); hands[f] = list(last)
def contains(a, b): return rect_overlap_ratio(b, a) >= 0.9
def region_at(f):
    rects = anch[f].get("box", [])
    if not rects: return None
    c = [r for r in rects if not any(r is not o and contains(r, o) for o in rects)] or rects
    return sub_rect(c[0], p.state_rel)
SEGS = [("拿鞋盒",52,182),("開鞋盒",204,290),("放防黴片",325,522),("放鞋子",542,692),("拿鞋盒",1171,1249),("開鞋盒",1336,1466),
        ("放防黴片",1490,1651),("放鞋子",1671,1904),("拿鞋盒",2357,2528),("開鞋盒",2548,2662),("放防黴片",2687,2868),("放鞋子",2893,3085)]
# timeline string: '.'=no box  'H'=hands inside  'o'=box free, one char per 5 frames, ticks every 100 frames
def timeline(f0, f1):
    s = ""
    for f in range(f0, f1, STEP):
        reg = region_at(f)
        s += "." if reg is None else ("H" if any_point_in_rect(hands[f], reg) else "o")
    return s
for f0, f1 in [(0, 1200), (1150, 2400), (2350, 3500)]:
    print(f"\n--- frames {f0}-{f1} (one char = 5 frames; '.' no box, 'H' hands in box, 'o' box free) ---")
    for a in range(f0, f1, 300):
        b = min(a + 300, f1)
        print(f"{a:5d} {timeline(a, b)}")
    print("segments:", [(n, s, e) for n, s, e in SEGS if s >= f0 and e <= f1])
# contact sheets of box crops
reader = FrameReader(p.video_path)
def sheet(name, frames, cols=7, tw=272, th=196):
    tiles = []
    for f in frames:
        f = int(round(f / STEP) * STEP)
        fr = reader.get_frame(f); reg = region_at(f)
        if reg is None:
            tile = np.zeros((th, tw, 3), np.uint8); cv2.putText(tile, "no box", (10, th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            crop = crop_roi(fr, reg)
            x, y, w, h = reg; sx, sy = tw / max(1, crop.shape[1]), th / max(1, crop.shape[0])
            tile = cv2.resize(crop, (tw, th))
            for hx, hy in hands[f]:
                if x <= hx <= x + w and y <= hy <= y + h:
                    cv2.circle(tile, (int((hx - x) * sx), int((hy - y) * sy)), 5, (0, 255, 0), -1)
        cv2.putText(tile, str(f), (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        tiles.append(tile)
    while len(tiles) % cols: tiles.append(np.zeros((th, tw, 3), np.uint8))
    rows_ = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    img = np.vstack(rows_); cv2.imwrite(os.path.join(OUT, name), img, [cv2.IMWRITE_JPEG_QUALITY, 85]); print("wrote", name, img.shape)
sheet("r1_open_to_card.jpg", list(range(280, 345, 5)) + list(range(505, 560, 5)))      # 開盒結束→放片開始, 放片結束→放鞋開始
sheet("r1_shoe_to_close.jpg", list(range(680, 785, 5)))                                   # 放鞋結束→蓋上
sheet("r2_card_to_shoe.jpg", list(range(1455, 1500, 5)) + list(range(1640, 1685, 5)))
sheet("r2_shoe_to_close.jpg", list(range(1895, 2000, 5)))
reader.release()
