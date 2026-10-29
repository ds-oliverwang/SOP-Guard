"""Evaluate item_recognizer on the desk video: cardboard model from the project's 空盒 examples, item examples from
round-1 blobs, then per judged frame find items; print per-run results + a sheet of masks/blobs."""
import os, sys, json, time, glob
import numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import torch  # noqa
from project_store import load_project
from scene import Detection, SceneTracker, rect_overlap_ratio
from core_detector import state_region, hands_in_inner, crop_roi, STATE_FRESH_SEC
from state_recognizer import Embedder
from item_recognizer import CardboardModel, ItemLibrary, item_blobs, interior_mask
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

empties = [cv2.imdecode(np.fromfile(fp, np.uint8), cv2.IMREAD_COLOR) for fp in glob.glob(os.path.join(D, "states", "空盒", "*.jpg"))]
model = CardboardModel.fit(empties)
print(f"cardboard model: angle {model.angle:.1f}°  chroma {model.chroma:.1f}  L {model.l_med:.0f}")
import shutil; TMP = os.path.join(ROOT, "cache", "_item_lib"); shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)
lib = ItemLibrary(TMP, embedder=Embedder(prefer_clip=True)); lib.model = model; lib.save()
# item examples from round 1 (naming by size/position rules, just for the experiment)
for f in (920, 950, 1000, 1020, 1090, 1105):
    img = crop(f); rects, fg, inside = item_blobs(img, model); area_in = max(1, cv2.countNonZero(inside))
    desc = []
    for (x, y, w, h) in rects:
        size = w * h / area_in
        if f in (1090, 1105):
            name = "鞋子" if size > 0.15 else None
        else:
            name = "黑色小物" if size < 0.12 and x > img.shape[1] * 0.45 else ("標籤" if size < 0.12 else None)
        desc.append(f"{name}:{size:.3f}@{x},{y},{w}x{h}")
        if name:
            lib.add_example(name, img[y:y + h, x:x + w], f, size)
    print(f"example frame {f}: {desc}")
print("examples:", lib.counts())
t0 = time.perf_counter(); found = {}; sheet = []
SHOW = (820, 920, 1000, 1090, 1900, 2120, 2790, 3690, 4520, 4675, 4865)
for f in range(0, N, STEP):
    reg, free = status[f]
    if not free:
        continue
    img = crop(f); hits = lib.find(img)
    found[f] = [(h.name, round(h.distance, 2), h.rect, round(h.size, 3)) for h in hits]
    if f in SHOW:
        rects, fg, inside = item_blobs(img, model); vis = img.copy()
        cnts, _ = cv2.findContours(inside, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(vis, cnts, -1, (255, 0, 255), 3)
        for (x, y, w, h) in rects:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 255), 2)
        for h_ in hits:
            x, y, w, h = h_.rect; cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 4)
            cv2.putText(vis, f"{h_.name} {h_.distance:.2f}", (x, max(22, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        a = cv2.resize(vis, (480, 420)); b = cv2.resize(cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR), (480, 420))
        cv2.putText(a, f"f{f}", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2); sheet += [a, b]
print(f"judged {len(found)} frames, {(time.perf_counter() - t0) / max(1, len(found)) * 1000:.0f} ms/frame")
rows_ = [np.hstack(sheet[i:i + 6]) for i in range(0, len(sheet), 6)]
w = max(r.shape[1] for r in rows_); rows_ = [np.hstack([r, np.zeros((r.shape[0], w - r.shape[1], 3), np.uint8)]) for r in rows_]
cv2.imwrite(os.path.join(OUT, "item_probe2.jpg"), np.vstack(rows_), [cv2.IMWRITE_JPEG_QUALITY, 80])
json.dump(found, open(os.path.join(OUT, "item_found2.json"), "w"), ensure_ascii=False)
print("\nper run: name×frames-with-hit/frames (best dist) | unnamed blobs per frame")
names = lib.order
for ri, rs in enumerate(runs, 1):
    for k, (f0, f1) in enumerate(rs, 1):
        fs = [f for f in range(f0, f1 + 1, STEP) if f in found]
        if not fs:
            continue
        cnt = {}; best = {}
        for f in fs:
            for nm in {h[0] for h in found[f]}:
                cnt[nm] = cnt.get(nm, 0) + 1; best[nm] = min(best.get(nm, 9), min(h[1] for h in found[f] if h[0] == nm))
        print(f"R{ri}-{k} f{f0}-{f1}: " + (", ".join(f"{nm}×{cnt[nm]}/{len(fs)}({best[nm]:.2f})" for nm in names if nm in cnt) or "—"))
reader.release()
