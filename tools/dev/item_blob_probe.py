"""Item model, approach B: 'anything that is not cardboard' inside the box → blobs → name each blob with CLIP nearest
neighbour against item examples (taken from round 1). Evaluate across all judged frames of the desk video."""
import os, sys, json, time
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
            nested = region is not None and max(rect_overlap_ratio(new, region), rect_overlap_ratio(region, new)) >= 0.9
            if not nested:
                moved_at = f
        region = new
    if f % STEP == 0:
        age = sc.age(p.state_anchor, f / FPS)
        status[f] = (region, region is not None and age is not None and age <= STATE_FRESH_SEC and (f - moved_at) >= settle and not hands_in_inner(last_hands, region))
reader = FrameReader(p.video_path)
def crop(f):
    return crop_roi(reader.get_frame(f), status[f][0])

# ---- cardboard colour model from 空盒 example crops: HSV, keep the central 60% (avoid box walls/outside)
import glob
empties = [cv2.imdecode(np.fromfile(fp, np.uint8), cv2.IMREAD_COLOR) for fp in glob.glob(os.path.join(D, "states", "空盒", "*.jpg"))]
hs = []
for im in empties:
    h, w = im.shape[:2]; c = im[int(h * .2):int(h * .8), int(w * .2):int(w * .8)]
    hs.append(cv2.cvtColor(c, cv2.COLOR_BGR2HSV).reshape(-1, 3))
hs = np.concatenate(hs).astype(np.float32)
H_lo, H_hi = np.percentile(hs[:, 0], [2, 98]); S_lo = np.percentile(hs[:, 1], 2); V_lo = np.percentile(hs[:, 2], 2)
print(f"cardboard model: H {H_lo:.0f}-{H_hi:.0f}  S>={S_lo:.0f}  V>={V_lo:.0f}  (from {len(empties)} empty-box crops)")

def blobs(img, min_frac=0.004, shrink=0.08):
    """foreground = not cardboard-coloured; ignore a border band (box walls / outside). returns [(x,y,w,h,mask)]"""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(np.float32), hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
    card = (H >= H_lo - 4) & (H <= H_hi + 4) & (S >= S_lo * 0.7) & (V >= V_lo * 0.7)
    m = (~card).astype(np.uint8) * 255
    b = int(min(w, h) * shrink); m[:b, :] = 0; m[-b:, :] = 0; m[:, :b] = 0; m[:, -b:] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)); m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw * ch >= min_frac * w * h:
            out.append((x, y, cw, ch))
    return sorted(out, key=lambda r: -r[2] * r[3]), m

emb = Embedder(prefer_clip=True)
# ---- item examples from round 1 frames (blobs that we can name by position/size)
def name_by_rule(f, rects):
    """round-1 naming rules just to build the example set."""
    out = {}
    for (x, y, w, h) in rects:
        area = w * h
        if f in (920, 950) and area < 0.12 * 1020 * 897:
            out.setdefault("黑色小物", []).append((x, y, w, h))
        if f in (1000, 1020):
            out.setdefault("標籤" if w * h < 0.12 * 1020 * 897 and x < 400 else "黑色小物", []).append((x, y, w, h))
        if f in (1090, 1105) and area > 0.15 * 1020 * 897:
            out.setdefault("鞋子", []).append((x, y, w, h))
    return out
examples = {}
for f in (920, 950, 1000, 1020, 1090, 1105):
    img = crop(f); rects, m = blobs(img)
    named = name_by_rule(f, rects)
    print(f"example frame {f}: blobs {rects} → {{k: len(v) for k, v in named.items()}}")
    for name, rs in named.items():
        for (x, y, w, h) in rs:
            examples.setdefault(name, []).append(emb.embed(img[y:y + h, x:x + w]))
print("examples:", {k: len(v) for k, v in examples.items()})
names = list(examples); X = {k: np.array(v) for k, v in examples.items()}
def classify(vec):
    d = {k: float(np.linalg.norm(X[k] - vec, axis=1).min()) for k in names}
    k = min(d, key=d.get); return k, d[k]

# ---- evaluate: per judged frame, blobs → names (distance ≤ 0.45)
t0 = time.perf_counter(); found = {}; sheet = []
for f in range(0, N, STEP):
    reg, free = status[f]
    if not free:
        continue
    img = crop(f); rects, m = blobs(img)
    res = []
    for (x, y, w, h) in rects[:4]:
        k, d = classify(emb.embed(img[y:y + h, x:x + w])); res.append((k, round(d, 2), (x, y, w, h)))
    found[f] = res
    if f in (820, 920, 1000, 1090, 1900, 2120, 4520, 4675, 4865):
        vis = img.copy()
        for k, d, (x, y, w, h) in res:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 3); cv2.putText(vis, f"{k} {d:.2f}", (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        a = cv2.resize(vis, (480, 420)); b = cv2.resize(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR), (480, 420))
        cv2.putText(a, f"f{f}", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2); sheet += [a, b]
print(f"judged {len(found)} frames, {(time.perf_counter() - t0) / max(1, len(found)) * 1000:.0f} ms/frame")
rows_ = [np.hstack(sheet[i:i + 6]) for i in range(0, len(sheet), 6)]
if rows_:
    w = max(r.shape[1] for r in rows_); rows_ = [np.hstack([r, np.zeros((r.shape[0], w - r.shape[1], 3), np.uint8)]) for r in rows_]
    cv2.imwrite(os.path.join(OUT, "item_blobs.jpg"), np.vstack(rows_), [cv2.IMWRITE_JPEG_QUALITY, 80])
print("\nper run: items found in ≥ half of the run's frames (name: best distance) | blobs/frame")
for ri, rs in enumerate(runs, 1):
    for k, (f0, f1) in enumerate(rs, 1):
        fs = [f for f in range(f0, f1 + 1, STEP) if f in found]
        if not fs:
            continue
        cnt = {}; best = {}
        for f in fs:
            seen = set()
            for nm, d, _ in found[f]:
                if d <= 0.45 and nm not in seen:
                    seen.add(nm); cnt[nm] = cnt.get(nm, 0) + 1; best[nm] = min(best.get(nm, 9), d)
        items = ", ".join(f"{nm}×{cnt[nm]}/{len(fs)}({best[nm]:.2f})" for nm in names if nm in cnt)
        print(f"R{ri}-{k} f{f0}-{f1}: {items or '—'}  | blobs {np.mean([len(found[f]) for f in fs]):.1f}")
reader.release()
