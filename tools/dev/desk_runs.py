"""From the project's trace: find 'box present, fresh, settled, hands out' runs, group them into rounds (gaps with no box),
and write one contact sheet per round showing the box crop at the middle of each run (so a human can name the state)."""
import os, sys, json
import numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
from project_store import load_project
from scene import Detection, SceneTracker, rect_overlap_ratio
from core_detector import state_region, hands_in_inner, crop_roi, STATE_FRESH_SEC
from video_recorder import FrameReader
D = sys.argv[1]; OUT = os.path.join(ROOT, "cache", "desk_runs"); os.makedirs(OUT, exist_ok=True)
p = load_project(D)
tr = json.load(open(os.path.join(D, "cache_trace.json"), encoding="utf-8"))
rows = {r["f"]: r for r in tr["rows"]}; N = tr["frames"]; FPS = tr["fps"]; STEP = 5
sc = SceneTracker(p.anchors, tuple(p.work_area) if p.work_area else None)
status = {}   # f -> (region or None, free: bool)
last_hands = []; region = None; moved_at = -10**9; settle = int(round(0.5 * FPS))
for f in range(N):
    r = rows.get(f)
    if r and "dets" in r:
        sc.update_detections([Detection(d[0], d[1], tuple(d[2:6])) for d in r["dets"]], f / FPS)
    if r and "hands" in r:
        last_hands = [tuple(h) for h in r["hands"]]
    anchors = sc.anchors_at(f / FPS)
    new = state_region(p, anchors)
    if new is None:
        region = None
    elif region is None or min(rect_overlap_ratio(new, region), rect_overlap_ratio(region, new)) < 0.6:
        nested = region is not None and max(rect_overlap_ratio(new, region), rect_overlap_ratio(region, new)) >= 0.9
        if not nested:
            moved_at = f
        region = new
    if f % STEP == 0:
        age = sc.age(p.state_anchor, f / FPS)
        free = region is not None and (age is not None and age <= STATE_FRESH_SEC) and (f - moved_at) >= settle and not hands_in_inner(last_hands, region)
        status[f] = (region, free)
# runs of consecutive free samples; rounds split where the box is absent for >= 2 s
runs = []; cur = []
for f in range(0, N, STEP):
    reg, free = status[f]
    if free:
        cur.append(f)
    elif cur:
        runs.append(cur); cur = []
if cur:
    runs.append(cur)
runs = [r for r in runs if len(r) >= 2]       # ≥ 0.33 s
rounds = []; last_end = None
for r in runs:
    if last_end is None or any(status[f][0] is None for f in range(last_end, r[0], STEP) if f in status) and (r[0] - last_end) >= 2 * FPS:
        rounds.append([])
    rounds[-1].append(r); last_end = r[-1]
reader = FrameReader(p.video_path)
tw, th = 300, 260
for ri, rs in enumerate(rounds, 1):
    tiles = []
    for k, r in enumerate(rs, 1):
        f = r[len(r) // 2]; reg = status[f][0]
        crop = crop_roi(reader.get_frame(f), reg); t = cv2.resize(crop, (tw, th))
        cv2.putText(t, f"R{ri}-{k} f{r[0]}-{r[-1]}", (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(t)
        print(f"round {ri} run {k}: frames {r[0]}-{r[-1]} ({len(r)} samples, {len(r) * STEP / FPS:.1f}s) region {[int(v) for v in reg]}")
    cols = 6
    while len(tiles) % cols:
        tiles.append(np.zeros((th, tw, 3), np.uint8))
    img = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(os.path.join(OUT, f"round{ri}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
print("rounds:", len(rounds), "->", OUT)
json.dump({"rounds": [[[r[0], r[-1]] for r in rs] for rs in rounds]}, open(os.path.join(OUT, "runs.json"), "w"))
reader.release()
