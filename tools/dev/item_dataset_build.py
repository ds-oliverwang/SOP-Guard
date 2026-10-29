"""Build the YOLO dataset the way the UI will: the user frames each item once per round (seed rect + name); the program
propagates the seed through the other hands-free frames of the same placement by template matching (same lighting,
box static) and stops when the match is weak (occluded / moved). Rounds 1-4 → train, round 5 → val."""
import os, sys, json, shutil, random
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
from item_core import *   # noqa
DS = os.path.join(ROOT, "cache", "item_ds"); shutil.rmtree(DS, ignore_errors=True)
for s in ("train", "val"):
    os.makedirs(os.path.join(DS, "images", s)); os.makedirs(os.path.join(DS, "labels", s))
CLASSES = ["黑色小物", "標籤", "鞋子"]
# (round window, seeds) — seed = (name, frame, rect in that frame's box crop)
ROUNDS = [((830, 1185), [("標籤", 1000, (255, 462, 340, 326)), ("黑色小物", 1000, (537, 470, 207, 217)), ("鞋子", 1100, (130, 80, 630, 600))]),
          ((1900, 2165), [("標籤", 2000, (382, 441, 180, 116)), ("黑色小物", 2000, (561, 441, 155, 162)), ("鞋子", 2120, (20, 80, 620, 700))]),
          ((2710, 3045), [("黑色小物", 2850, (290, 280, 195, 165)), ("標籤", 2850, (265, 444, 300, 164)), ("鞋子", 2960, (0, 72, 380, 497))]),
          ((3550, 3980), [("鞋子", 3850, (0, 42, 498, 493))]),
          ((4520, 4950), [("黑色小物", 4700, (232, 194, 155, 116)), ("標籤", 4700, (483, 272, 165, 135)), ("鞋子", 4865, (171, 92, 564, 596))])]
def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def track(tpl, g, scales=(0.9, 1.0, 1.1)):
    best = (-1.0, None)
    for s in scales:
        t = cv2.resize(tpl, None, fx=s, fy=s) if s != 1.0 else tpl
        if t.shape[0] >= g.shape[0] or t.shape[1] >= g.shape[1]:
            continue
        res = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED); _, mx, _, loc = cv2.minMaxLoc(res)
        if mx > best[0]:
            best = (float(mx), (loc[0], loc[1], t.shape[1], t.shape[0]))
    return best
labels = {}; seed_vis = []; stats = {"train": {}, "val": {}}
for ri, ((f0, f1), seeds) in enumerate(ROUNDS, 1):
    split = "val" if ri == 5 else "train"
    tpls = []
    ref_g = gray(crop(f0))                     # 這一輪的空盒參考：物品的樣板在空盒上也配得到就不算（大塊深色鞋子會配到陰影）
    for name, sf, (x, y, w, h) in seeds:
        sf5 = sf - sf % STEP
        # seed frame may not be a sampled frame: crop with the nearest sampled region
        reg = status[sf5][0] if status[sf5][0] is not None else status[sf5 - 5][0]
        img = crop_roi(reader.get_frame(sf), reg)
        tpl = gray(img)[y:y + h, x:x + w]
        ref_score = track(tpl, ref_g)[0]
        tpls.append((name, tpl, ref_score))
        print(f"round {ri} seed {name} f{sf}: score on empty reference {ref_score:.2f}")
        vis = img.copy(); cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 4); cv2.putText(vis, f"{name} f{sf}", (x, max(30, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        seed_vis.append(cv2.resize(vis, (400, 350)))
    for f in range(f0, f1 + 1, STEP):
        reg, free = status.get(f, (None, False))
        if not free:
            continue
        img = crop(f); g = gray(img); found = []
        for name, tpl, ref_score in tpls:
            score, r = track(tpl, g)
            if r is not None and score >= 0.75 and score - ref_score >= 0.12:
                found.append((name, score, r))
        if not found:
            continue
        labels[f] = (split, img, found)
        for name, _, _ in found:
            stats[split][name] = stats[split].get(name, 0) + 1
for f, (split, img, found) in labels.items():
    H, Wd = img.shape[:2]; nm = f"f{f:05d}"
    cv2.imwrite(os.path.join(DS, "images", split, nm + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(os.path.join(DS, "labels", split, nm + ".txt"), "w") as fh:
        for name, score, (x, y, w, h) in found:
            fh.write(f"{CLASSES.index(name)} {(x + w / 2) / Wd:.6f} {(y + h / 2) / H:.6f} {w / Wd:.6f} {h / H:.6f}\n")
print("images:", {s: sum(1 for v in labels.values() if v[0] == s) for s in ("train", "val")}, "labels:", stats)
random.seed(1); sample = random.sample(sorted(labels), min(12, len(labels)))
tiles = list(seed_vis)
for f in sample:
    split, img, found = labels[f]; vis = img.copy()
    for name, score, (x, y, w, h) in found:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3); cv2.putText(vis, f"{name} {score:.2f}", (x, max(26, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    t = cv2.resize(vis, (400, 350)); cv2.putText(t, f"f{f} {split}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2); tiles.append(t)
while len(tiles) % 6:
    tiles.append(np.zeros((350, 400, 3), np.uint8))
cv2.imwrite(os.path.join(OUT, "item_dataset.jpg"), np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
with open(os.path.join(DS, "data.yaml"), "w", encoding="utf-8") as fh:
    fh.write(f"path: {DS}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))
reader.release()
