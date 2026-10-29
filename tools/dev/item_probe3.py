"""Item approach E: align each box crop to an EMPTY-box reference (ECC euclidean on downscaled gray), then compare
EDGE maps (robust to shading) → 'new edges' blobs = things that were put in. Evaluate blob quality per run first."""
import os, sys, json, time
import numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
from project_store import load_project
from scene import Detection, SceneTracker, rect_overlap_ratio
from core_detector import state_region, hands_in_inner, crop_roi, STATE_FRESH_SEC
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

W = 384   # working width
def prep(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape; s = W / w
    g = cv2.resize(g, (W, max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
    return g, s
def edges(g):
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)
REF_F = 830
ref_img = crop(REF_F); ref_g, _ = prep(ref_img)
H0, W0 = ref_g.shape
ref_e = edges(cv2.GaussianBlur(ref_g, (3, 3), 0))
ref_e_d = cv2.dilate(ref_e, np.ones((7, 7), np.uint8))          # tolerance for small misalignment
def align(g):
    """euclidean ECC: warp g onto the reference grid; returns (warped, ok, cc)."""
    gs = cv2.resize(g, (W0, H0)) if g.shape != (H0, W0) else g
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(ref_g, gs, warp, cv2.MOTION_EUCLIDEAN,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-4), None, 5)
    except cv2.error:
        return gs, False, 0.0, warp
    out = cv2.warpAffine(gs, warp, (W0, H0), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    return out, True, float(cc), warp
def new_blobs(img, thr=60, min_frac=0.004):
    g, s = prep(img)
    a, ok, cc, warp = align(g)
    e = edges(cv2.GaussianBlur(a, (3, 3), 0))
    diff = np.clip(e - ref_e_d, 0, None)
    m = (diff > thr).astype(np.uint8) * 255
    b = int(0.06 * min(W0, H0)); m[:b, :] = 0; m[-b:, :] = 0; m[:, :b] = 0; m[:, -b:] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_frac * W0 * H0:
            rects.append((x, y, w, h, int(cv2.countNonZero(m[y:y + h, x:x + w]))))
    rects.sort(key=lambda r: -r[2] * r[3])
    return rects, m, a, ok, cc
t0 = time.perf_counter(); res = {}; sheet = []
SHOW = (820, 920, 1000, 1090, 1900, 2120, 2790, 3690, 4520, 4675, 4865)
for f in range(0, N, STEP):
    reg, free = status[f]
    if not free:
        continue
    img = crop(f); rects, m, a, ok, cc = new_blobs(img)
    res[f] = {"ok": ok, "cc": round(cc, 3), "rects": rects[:5], "area": round(float(cv2.countNonZero(m)) / (W0 * H0), 4)}
    if f in SHOW:
        vis = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        for (x, y, w, h, n) in rects[:5]:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(vis, f"f{f} cc={cc:.2f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        sheet += [cv2.resize(vis, (384, 336)), cv2.resize(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR), (384, 336))]
print(f"judged {len(res)} frames, {(time.perf_counter() - t0) / max(1, len(res)) * 1000:.0f} ms/frame; ref frame {REF_F} size {W0}x{H0}")
rows_ = [np.hstack(sheet[i:i + 6]) for i in range(0, len(sheet), 6)]
w = max(r.shape[1] for r in rows_); rows_ = [np.hstack([r, np.zeros((r.shape[0], w - r.shape[1], 3), np.uint8)]) for r in rows_]
cv2.imwrite(os.path.join(OUT, "item_probe3.jpg"), np.vstack(rows_), [cv2.IMWRITE_JPEG_QUALITY, 80])
print("\nper run: median new-edge area %, median #blobs, ECC ok ratio, median cc")
for ri, rs in enumerate(runs, 1):
    for k, (f0, f1) in enumerate(rs, 1):
        fs = [f for f in range(f0, f1 + 1, STEP) if f in res]
        if not fs:
            continue
        print(f"R{ri}-{k} f{f0}-{f1}: area {np.median([res[f]['area'] for f in fs]) * 100:5.1f}%  blobs {np.median([len(res[f]['rects']) for f in fs]):.0f}  "
              f"ok {np.mean([res[f]['ok'] for f in fs]):.2f}  cc {np.median([res[f]['cc'] for f in fs]):.2f}")
reader.release()
