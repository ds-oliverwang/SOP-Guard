"""Shared pieces for the item-model experiments on the desk video: trace replay (regions / hands-free status),
box crops, and the aligned edge-difference proposal engine (approach C)."""
import os, sys, json
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
REFS = [830, 1900, 2710, 3550, 4520]        # 每輪「空盒」參考格（實際流程由使用者點一次）
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

W = 384
def prep(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); h, w = g.shape; s = W / w
    g = cv2.resize(g, (W, max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return cv2.createCLAHE(2.0, (8, 8)).apply(g), s
def edges(g):
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3); return cv2.magnitude(gx, gy)

class Ref:
    """空盒參考：之後的影格先對齊到它，再看多出來的邊緣 → 新放進去的東西。"""
    def __init__(self, img):
        self.g, self.s = prep(img); self.H, self.W = self.g.shape
        self.e_d = cv2.dilate(edges(cv2.GaussianBlur(self.g, (3, 3), 0)), np.ones((7, 7), np.uint8))
    def align(self, img):
        g, s = prep(img); gs = cv2.resize(g, (self.W, self.H)) if g.shape != (self.H, self.W) else g
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            cc, warp = cv2.findTransformECC(self.g, gs, warp, cv2.MOTION_EUCLIDEAN, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-4), None, 5)
        except cv2.error:
            return None, 0.0, warp
        a = cv2.warpAffine(gs, warp, (self.W, self.H), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        return a, float(cc), warp
    def proposals(self, img, thr=60, min_frac=0.004):
        a, cc, warp = self.align(img)
        if a is None or cc < 0.6:
            return [], cc
        m = (np.clip(edges(cv2.GaussianBlur(a, (3, 3), 0)) - self.e_d, 0, None) > thr).astype(np.uint8) * 255
        b = int(0.06 * min(self.W, self.H)); m[:b, :] = 0; m[-b:, :] = 0; m[:, :b] = 0; m[:, -b:] = 0
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)); m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); fine = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w * h >= min_frac * self.W * self.H:
                fine.append((x, y, w, h))
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
        sx = img.shape[1] / self.W; sy = img.shape[0] / self.H; rects = []
        for (x, y, w, h) in cands:
            pts = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], np.float32)
            pts2 = (warp[:, :2] @ pts.T).T + warp[:, 2]
            x1, y1 = max(0, pts2[:, 0].min()), max(0, pts2[:, 1].min()); x2, y2 = min(self.W, pts2[:, 0].max()), min(self.H, pts2[:, 1].max())
            if x2 > x1 and y2 > y1:
                rects.append((int(x1 * sx), int(y1 * sy), int((x2 - x1) * sx), int((y2 - y1) * sy)))
        return sorted(rects, key=lambda r: -r[2] * r[3]), cc
