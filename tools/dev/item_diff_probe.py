"""Feasibility probe for the user's 'item' model: can we automatically find what was put into the box by differencing
the box interior before/after (same round, box not moved)?  Uses the example crops already stored in the project."""
import os, sys, json
import numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(ROOT)
D = os.path.join(ROOT, "sop_projects", "桌面狀態測試", "states")
ex = json.load(open(os.path.join(D, "states.json"), encoding="utf-8"))["examples"]
def load(rel):
    return cv2.imdecode(np.fromfile(os.path.join(D, rel), np.uint8), cv2.IMREAD_COLOR)
by = {}
for e in ex:
    by.setdefault(e["state"], []).append(e)
# round 1: 空盒 (frames 820~840) vs 已放標籤 (920~980) vs 已放鞋子 (1090~1105)
def pick(state, lo, hi):
    return next(e for e in by[state] if lo <= e["frame"] <= hi)
pairs = [("空盒→已放標籤", pick("空盒", 800, 860), pick("已放標籤", 900, 1000)),
         ("已放標籤→已放鞋子", pick("已放標籤", 900, 1000), pick("已放鞋子", 1050, 1130)),
         ("第5輪 空盒→已放標籤", pick("空盒", 4500, 4620), pick("已放標籤", 4630, 4770))]
tiles = []
for name, a, b in pairs:
    A, B = load(a["file"]), load(b["file"])
    h, w = 540, 640
    A = cv2.resize(A, (w, h)); B = cv2.resize(B, (w, h))
    ga = cv2.GaussianBlur(cv2.cvtColor(A, cv2.COLOR_BGR2GRAY), (7, 7), 0); gb = cv2.GaussianBlur(cv2.cvtColor(B, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    d = cv2.absdiff(ga, gb)
    _, m = cv2.threshold(d, 35, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis = B.copy(); found = []
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw * ch < 0.003 * w * h:
            continue
        found.append((x, y, cw, ch)); cv2.rectangle(vis, (x, y), (x + cw, y + ch), (0, 255, 255), 2)
    print(f"{name}: frames {a['frame']}→{b['frame']}  changed blobs {len(found)} {found}  changed area {m.mean() / 255:.1%}")
    for img, t in ((A, f"{name} 前 f{a['frame']}"), (B, f"後 f{b['frame']}"), (cv2.cvtColor(m, cv2.COLOR_GRAY2BGR), "差異"), (vis, "自動框出的新物品")):
        img = img.copy(); cv2.putText(img, t, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2); tiles.append(img)
rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
out = os.path.join(ROOT, "cache", "desk_overview", "item_diff.jpg"); cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85]); print("wrote", out)
