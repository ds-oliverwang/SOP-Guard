"""Build a YOLO dataset from the desk video the way the UI will: per placement, reference = first settled hands-free
frame; proposals = aligned edge-diff blobs (approach C); naming = CLIP nearest neighbour to round-1 item crops
(stands in for the user's naming). Rounds 1-4 → train, round 5 → val. Writes a contact sheet for sanity checking."""
import os, sys, json, time, shutil
import numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import torch  # noqa
from project_store import load_project
from scene import Detection, SceneTracker, rect_overlap_ratio
from core_detector import state_region, hands_in_inner, crop_roi, STATE_FRESH_SEC
from state_recognizer import Embedder
from video_recorder import FrameReader
D = os.path.join(ROOT, "sop_projects", "桌面狀態測試"); OUT = os.path.join(ROOT, "cache", "desk_overview")
DS = os.path.join(ROOT, "cache", "item_ds"); shutil.rmtree(DS, ignore_errors=True)
for s in ("train", "val"):
    os.makedirs(os.path.join(DS, "images", s)); os.makedirs(os.path.join(DS, "labels", s))
p = load_project(D)
tr = json.load(open(os.path.join(D, "cache_trace.json"), encoding="utf-8"))
rows = {r["f"]: r for r in tr["rows"]}; N = tr["frames"]; FPS = tr["fps"]; STEP = 5
sc = SceneTracker(p.anchors, tuple(p.work_area) if p.work_area else None)
status = {}; last_hands = []; region = None; moved_at = -10**9; settle = int(round(0.5 * FPS)); placement = 0; place_of = {}
for f in range(N):
    r = rows.get(f)
    if r and "dets" in r:
        sc.update_detections([Detection(d[0], d[1], tuple(d[2:6])) for d in r["dets"]], f / FPS)
    if r and "hands" in r:
        last_hands = [tuple(h) for h in r["hands"]]
    new = state_region(p, sc.anchors_at(f / FPS))
    if new is None:
        if region is not None:
            placement += 1
        region = None
    else:
        if region is None or min(rect_overlap_ratio(new, region), rect_overlap_ratio(region, new)) < 0.6:
            if not (region is not None and max(rect_overlap_ratio(new, region), rect_overlap_ratio(region, new)) >= 0.9):
                moved_at = f; placement += 1
        region = new
    if f % STEP == 0:
        age = sc.age(p.state_anchor, f / FPS)
        free = region is not None and age is not None and age <= STATE_FRESH_SEC and (f - moved_at) >= settle and not hands_in_inner(last_hands, region)
        status[f] = (region, free); place_of[f] = placement
reader = FrameReader(p.video_path)
def crop(f):
    return crop_roi(reader.get_frame(f), status[f][0])
W = 384
def prep(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); h, w = g.shape; s = W / w
    g = cv2.resize(g, (W, max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return cv2.createCLAHE(2.0, (8, 8)).apply(g), s
def edges(g):
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3); return cv2.magnitude(gx, gy)
class Ref:
    def __init__(self, img):
        self.g, self.s = prep(img); self.H, self.W = self.g.shape
        self.e_d = cv2.dilate(edges(cv2.GaussianBlur(self.g, (3, 3), 0)), np.ones((7, 7), np.uint8))
    def proposals(self, img, thr=60, min_frac=0.004):
        g, s = prep(img); gs = cv2.resize(g, (self.W, self.H)) if g.shape != (self.H, self.W) else g
        warp = np.eye(2, 3, dtype=np.float32); ok = True
        try:
            cc, warp = cv2.findTransformECC(self.g, gs, warp, cv2.MOTION_EUCLIDEAN, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-4), None, 5)
        except cv2.error:
            ok = False; cc = 0.0
        if not ok or cc < 0.6:
            return [], cc
        a = cv2.warpAffine(gs, warp, (self.W, self.H), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        m = (np.clip(edges(cv2.GaussianBlur(a, (3, 3), 0)) - self.e_d, 0, None) > thr).astype(np.uint8) * 255
        b = int(0.06 * min(self.W, self.H)); m[:b, :] = 0; m[-b:, :] = 0; m[:, :b] = 0; m[:, -b:] = 0
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)); m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); fine = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w * h >= min_frac * self.W * self.H:
                fine.append((x, y, w, h))
        # candidates = fine blobs + clusters of blobs closer than 4% of width (a shoe breaks into several edge blobs)
        gap = 0.04 * self.W; clusters = []
        for r in sorted(fine, key=lambda r: -r[2] * r[3]):
            for cl in clusters:
                cx, cy, cw, ch = cl["box"]
                if r[0] < cx + cw + gap and cx < r[0] + r[2] + gap and r[1] < cy + ch + gap and cy < r[1] + r[3] + gap:
                    x1, y1 = min(cx, r[0]), min(cy, r[1]); x2, y2 = max(cx + cw, r[0] + r[2]), max(cy + ch, r[1] + r[3])
                    cl["box"] = (x1, y1, x2 - x1, y2 - y1); cl["n"] += 1; break
            else:
                clusters.append({"box": r, "n": 1})
        cands = fine + [cl["box"] for cl in clusters if cl["n"] > 1]
        # rects are on the reference grid (current image was warped onto it); map back: warp maps ref→current
        sx = img.shape[1] / self.W; sy = img.shape[0] / self.H; rects = []
        for (x, y, w, h) in cands:
            pts = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], np.float32)
            pts2 = (warp[:, :2] @ pts.T).T + warp[:, 2]
            x1, y1 = pts2.min(0); x2, y2 = pts2.max(0)
            x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(self.W, x2), min(self.H, y2)
            if x2 > x1 and y2 > y1:
                rects.append((int(x1 * sx), int(y1 * sy), int((x2 - x1) * sx), int((y2 - y1) * sy)))
        return sorted(rects, key=lambda r: -r[2] * r[3])[:8], cc
emb = Embedder(prefer_clip=True)
# round-1 item example crops (stand-in for user naming)
EX = {"黑色小物": [(920, (541, 463, 204, 223))], "標籤": [(1000, (258, 466, 335, 320))], "鞋子": [(1090, (131, 78, 633, 606))]}
X = {}
for name, lst in EX.items():
    for f, (x, y, w, h) in lst:
        X[name] = np.array([emb.embed(crop(f)[y:y + h, x:x + w])])
def name_blob(img, r):
    x, y, w, h = r; v = emb.embed(img[y:y + h, x:x + w]); best = (None, 9.0)
    for n, A in X.items():
        d = float(np.linalg.norm(A - v, axis=1).min())
        if d < best[1]:
            best = (n, d)
    return best
CLASSES = ["黑色小物", "標籤", "鞋子"]
# reference = the empty-box frame the user would click once per round (stand-in: hand-picked frames)
REFS = [830, 1900, 2710, 3550, 4520]
refs = {r: Ref(crop(r)) for r in REFS}
count = {"train": 0, "val": 0}; sheet = []; stats = {}
for f in range(0, N, STEP):
    reg, free = status[f]
    if not free or f < REFS[0]:
        continue
    pl = max(r for r in REFS if r <= f); img = crop(f)
    if f == pl:
        continue
    rects, cc = refs[pl].proposals(img)
    labels = []
    for r in rects:
        n, d = name_blob(img, r)
        if n is not None and d <= 0.35:
            # drop candidates that overlap an already-kept one (fine blob vs its cluster): keep the closer match
            dup = [i for i, (n2, d2, r2) in enumerate(labels) if rect_overlap_ratio(r, r2) > 0.5 or rect_overlap_ratio(r2, r) > 0.5]
            if dup and labels[dup[0]][1] <= d:
                continue
            for i in sorted(dup, reverse=True):
                labels.pop(i)
            labels.append((n, d, r))
    if f in (820, 900, 920, 1000, 1090, 4520, 4675):
        print(f"debug f{f}: placement {pl} cc {cc:.2f} rects {rects} named {[(n, round(d, 2)) for n, d, _ in labels]}")
    if not labels:
        continue
    split = "val" if f >= 4400 else "train"           # round 5 (frames ≥ 4400) held out
    H, Wd = img.shape[:2]; name = f"f{f:05d}"
    cv2.imwrite(os.path.join(DS, "images", split, name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(os.path.join(DS, "labels", split, name + ".txt"), "w") as fh:
        for n, d, (x, y, w, h) in labels:
            fh.write(f"{CLASSES.index(n)} {(x + w / 2) / Wd:.6f} {(y + h / 2) / H:.6f} {w / Wd:.6f} {h / H:.6f}\n")
            stats.setdefault(split, {}).setdefault(n, 0); stats[split][n] += 1
    count[split] += 1
    if len(sheet) < 36 and (f % 50 == 0):
        vis = img.copy()
        for n, d, (x, y, w, h) in labels:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 3); cv2.putText(vis, f"{n} {d:.2f}", (x, max(22, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        t = cv2.resize(vis, (400, 350)); cv2.putText(t, f"f{f} p{pl} {split}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2); sheet.append(t)
print("images:", count, "labels:", stats, "placements:", len(refs))
if sheet:
    while len(sheet) % 6:
        sheet.append(np.zeros((350, 400, 3), np.uint8))
    cv2.imwrite(os.path.join(OUT, "item_dataset.jpg"), np.vstack([np.hstack(sheet[i:i + 6]) for i in range(0, len(sheet), 6)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
with open(os.path.join(DS, "data.yaml"), "w", encoding="utf-8") as fh:
    fh.write(f"path: {DS}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))
reader.release()
