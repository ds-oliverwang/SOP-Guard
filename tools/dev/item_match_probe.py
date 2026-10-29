"""Feasibility of the ITEM model on the user's desk video: templates for 3 items taken from round 1, then for every
judged frame (box fresh+settled, hands out) find each item inside the box crop with rotation/scale-tolerant template
matching. Print per-run max scores so present/absent separation can be judged against the known timeline."""
import os, sys, json, time
import numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
from project_store import load_project
from scene import Detection, SceneTracker, rect_overlap_ratio
from core_detector import state_region, hands_in_inner, crop_roi, STATE_FRESH_SEC
from video_recorder import FrameReader
D = os.path.join(ROOT, "sop_projects", "桌面狀態測試")
p = load_project(D)
tr = json.load(open(os.path.join(D, "cache_trace.json"), encoding="utf-8"))
rows = {r["f"]: r for r in tr["rows"]}; N = tr["frames"]; FPS = tr["fps"]; STEP = 5
runs = json.load(open(os.path.join(ROOT, "cache", "desk_runs", "runs.json")))["rounds"]
# replay (same as desk_runs.py) → per sampled frame: region, free
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
    elif region is None or min(rect_overlap_ratio(new, region), rect_overlap_ratio(region, new)) < 0.6:
        nested = region is not None and max(rect_overlap_ratio(new, region), rect_overlap_ratio(region, new)) >= 0.9
        if not nested:
            moved_at = f
        region = new
    else:
        region = new
    if f % STEP == 0:
        age = sc.age(p.state_anchor, f / FPS)
        free = region is not None and age is not None and age <= STATE_FRESH_SEC and (f - moved_at) >= settle and not hands_in_inner(last_hands, region)
        status[f] = (region, free)
reader = FrameReader(p.video_path)
def gray_crop(f, reg):
    return cv2.cvtColor(crop_roi(reader.get_frame(f), reg), cv2.COLOR_BGR2GRAY)

# ---- templates (frame coords, round 1). black item: from diff blob; label: diff f920→f1000 inside the static run; shoe: hand rect
def blob_rect(fa, fb, reg, min_frac=0.003, pick="largest"):
    A = cv2.GaussianBlur(gray_crop(fa, reg), (7, 7), 0); B = cv2.GaussianBlur(gray_crop(fb, reg), (7, 7), 0)
    _, m = cv2.threshold(cv2.absdiff(A, B), 35, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = m.shape; cands = []
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw * ch < min_frac * w * h or x == 0 or y == 0 or x + cw >= w or y + ch >= h:
            continue
        cands.append((cw * ch, (x, y, cw, ch)))
    cands.sort(reverse=True)
    return [c[1] for c in cands]
reg1 = status[920][0]
black = blob_rect(820 if status[820][0] else 835, 920, reg1)[0]
print("black item rect in crop:", black, "region", [int(v) for v in reg1])
reg2 = status[1000][0]
lab_cands = blob_rect(920, 1000, reg2)
print("label candidates:", lab_cands)
label = lab_cands[0]
reg3 = status[1090][0]
shoe = (int(0.235 * reg3[2]), int(0.17 * reg3[3]), int(0.52 * reg3[2]), int(0.52 * reg3[3]))   # 由 item_diff.jpg 讀出的概略位置
templates = {"黑色小物": (920, reg1, black), "標籤": (1000, reg2, label), "鞋子": (1090, reg3, shoe)}
T = {}
for name, (f, reg, (x, y, w, h)) in templates.items():
    g = gray_crop(f, reg)[y:y + h, x:x + w]
    T[name] = g; print(f"template {name}: frame {f} size {w}x{h}")
    cv2.imwrite(os.path.join(ROOT, "cache", "desk_overview", f"tpl_{name}.jpg"), g)

SCALE = 0.5
def match(g, tpl):
    """best TM_CCOEFF_NORMED over rotations/scales; both downscaled by SCALE."""
    gs = cv2.resize(g, None, fx=SCALE, fy=SCALE); best = -1.0
    for s in (0.85, 1.0, 1.15):
        t0 = cv2.resize(tpl, None, fx=SCALE * s, fy=SCALE * s)
        for ang in (-12, -6, 0, 6, 12):
            if ang:
                M = cv2.getRotationMatrix2D((t0.shape[1] / 2, t0.shape[0] / 2), ang, 1.0)
                t = cv2.warpAffine(t0, M, (t0.shape[1], t0.shape[0]), borderMode=cv2.BORDER_REPLICATE)
            else:
                t = t0
            if t.shape[0] >= gs.shape[0] or t.shape[1] >= gs.shape[1]:
                continue
            res = cv2.matchTemplate(gs, t, cv2.TM_CCOEFF_NORMED)
            best = max(best, float(res.max()))
    return best
t0 = time.perf_counter(); scores = {}
for f in range(0, N, STEP):
    reg, free = status[f]
    if not free:
        continue
    g = gray_crop(f, reg)
    scores[f] = {name: match(g, tpl) for name, tpl in T.items()}
print(f"judged frames {len(scores)}  {(time.perf_counter() - t0) / max(1, len(scores)) * 1000:.0f} ms per frame (3 items)")
json.dump(scores, open(os.path.join(ROOT, "cache", "desk_overview", "item_scores.json"), "w"))
print("\nper run max score (round-run frames): 黑色小物 / 標籤 / 鞋子")
for ri, rs in enumerate(runs, 1):
    for k, (f0, f1) in enumerate(rs, 1):
        ss = [scores[f] for f in range(f0, f1 + 1, STEP) if f in scores]
        if not ss:
            continue
        mx = {n: max(s[n] for s in ss) for n in T}
        md = {n: float(np.median([s[n] for s in ss])) for n in T}
        print(f"R{ri}-{k} f{f0}-{f1}: max {mx['黑色小物']:.2f}/{mx['標籤']:.2f}/{mx['鞋子']:.2f}   median {md['黑色小物']:.2f}/{md['標籤']:.2f}/{md['鞋子']:.2f}")
reader.release()
