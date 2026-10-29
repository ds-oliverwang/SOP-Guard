"""Item approach D: no segmentation. Slide a fixed grid of windows over the box crop, CLIP-embed each window, and for
each item compare 'distance to item examples' vs 'distance to empty-box windows'. Item present = some window is
closer to the item than to empty by a margin. Examples from round 1; evaluate all judged frames."""
import os, sys, json, time, glob
import numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import torch  # noqa
from project_store import load_project
from scene import Detection, SceneTracker, rect_overlap_ratio
from core_detector import state_region, hands_in_inner, crop_roi, STATE_FRESH_SEC
from state_recognizer import Embedder
from video_recorder import FrameReader
D = os.path.join(ROOT, "sop_projects", "桌面狀態測試"); OUT = os.path.join(ROOT, "cache", "desk_overview")
p = load_project(D)
tr = json.load(open(os.path.join(D, "cache_trace.json"), encoding="utf-8"))
rows = {r["f"]: r for r in tr["rows"]}; N = tr["frames"]; FPS = tr["fps"]; STEP = 5
runs = json.load(open(os.path.join(ROOT, "cache", "desk_runs", "runs.json")))["rounds"]
sc = SceneTracker(p.anchors, tuple(p.work_area) if p.work_area else None)
status = {}; last_hands = []; region = None; moved_at = -10**9; settle = int(round(0.5 * FPS))
for f in range(N):
    r = rows.get(f)
    if r and "dets" in r:
        sc.update_detections([Detection(d[0], d[1], tuple(d[2:6])) for d in r["dets"]], f / FPS)
    if r and "hands" in r:
        last_hands = [tuple(h) for h in r["hands"]]
    new = state_region(p, sc.anchors_at(f / FPS))
    if new is None:
        region = None
    else:
        if region is None or min(rect_overlap_ratio(new, region), rect_overlap_ratio(region, new)) < 0.6:
            if not (region is not None and max(rect_overlap_ratio(new, region), rect_overlap_ratio(region, new)) >= 0.9):
                moved_at = f
        region = new
    if f % STEP == 0:
        age = sc.age(p.state_anchor, f / FPS)
        status[f] = (region, region is not None and age is not None and age <= STATE_FRESH_SEC and (f - moved_at) >= settle and not hands_in_inner(last_hands, region))
reader = FrameReader(p.video_path)
def crop(f):
    return crop_roi(reader.get_frame(f), status[f][0])
emb = Embedder(prefer_clip=True)

def windows(img):
    """(rect, crop) list: 3x3 grid of half-size windows (stride 1/4) + 2x2 of 2/3-size + whole."""
    h, w = img.shape[:2]; out = []
    for frac, n in ((0.5, 3), (0.667, 2)):
        ww, wh = int(w * frac), int(h * frac)
        for i in range(n):
            for j in range(n):
                x = int((w - ww) * j / max(1, n - 1)); y = int((h - wh) * i / max(1, n - 1))
                out.append(((x, y, ww, wh), img[y:y + wh, x:x + ww]))
    out.append(((0, 0, w, h), img))
    return out
def embed_windows(img):
    ws = windows(img)
    return [(r, emb.embed(c)) for r, c in ws]

# ---- empty windows (negatives) from the project's 空盒 example crops
empties = []
for fp in glob.glob(os.path.join(D, "states", "空盒", "*.jpg")):
    im = cv2.imdecode(np.fromfile(fp, np.uint8), cv2.IMREAD_COLOR)
    empties += [v for _, v in embed_windows(im)]
E = np.array(empties); print("empty windows:", len(E))
# ---- item examples: windows of round-1 frames that overlap the known item rects (crop coords)
ITEM_RECTS = {"黑色小物": [(920, (541, 463, 204, 223)), (950, (541, 463, 204, 223)), (1000, (555, 480, 190, 200))],
              "標籤": [(1000, (258, 466, 335, 320)), (1020, (258, 466, 335, 320))],
              "鞋子": [(1090, (131, 78, 633, 606)), (1105, (131, 78, 633, 606))]}
def overlap(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    return max(0, x2 - x1) * max(0, y2 - y1) / max(1, b[2] * b[3])
items = {}
for name, lst in ITEM_RECTS.items():
    for f, rect in lst:
        img = crop(f)
        for r, v in embed_windows(img):
            if overlap(r, rect) >= 0.8:           # window contains ≥80% of the item
                items.setdefault(name, []).append((v, r[2] * r[3] / (img.shape[0] * img.shape[1])))
print("item example windows:", {k: len(v) for k, v in items.items()})
I = {k: np.array([v for v, _ in lst]) for k, lst in items.items()}

t0 = time.perf_counter(); scores = {}
for f in range(0, N, STEP):
    reg, free = status[f]
    if not free:
        continue
    ws = embed_windows(crop(f)); sc_ = {}
    for name, X in I.items():
        best = -9.0
        for r, v in ws:
            d_item = float(np.linalg.norm(X - v, axis=1).min()); d_empty = float(np.linalg.norm(E - v, axis=1).min())
            best = max(best, d_empty - d_item)
        sc_[name] = round(best, 3)
    scores[f] = sc_
print(f"judged {len(scores)} frames, {(time.perf_counter() - t0) / max(1, len(scores)) * 1000:.0f} ms/frame (14 windows)")
json.dump(scores, open(os.path.join(OUT, "item_scores4.json"), "w"))
names = list(I)
print("\nper run: max over frames of (d_empty - d_item), per item  [>0 = closer to item than to empty]")
for ri, rs in enumerate(runs, 1):
    for k, (f0, f1) in enumerate(rs, 1):
        fs = [f for f in range(f0, f1 + 1, STEP) if f in scores]
        if not fs:
            continue
        mx = {n: max(scores[f][n] for f in fs) for n in names}; md = {n: float(np.median([scores[f][n] for f in fs])) for n in names}
        print(f"R{ri}-{k} f{f0}-{f1}: " + "  ".join(f"{n} max {mx[n]:+.2f} med {md[n]:+.2f}" for n in names))
reader.release()
