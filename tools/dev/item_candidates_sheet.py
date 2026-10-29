"""Emulate the user's labelling step: for sampled frames of each placement, draw the auto-proposed candidate boxes
(numbered) on contact sheets so a human can assign names. Also dumps candidates.json for the next step.
Shared core (replay/proposals) lives in item_core.py."""
import os, sys, json
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
from item_core import *   # noqa: status, crop, Ref, REFS, STEP, N, OUT, reader

refs = {r: Ref(crop(r)) for r in REFS}
cands = {}; sheets = {}
for f in range(REFS[0], N, STEP):
    reg, free = status[f]
    if not free or f % 25:
        continue
    pl = max(r for r in REFS if r <= f)
    if f == pl:
        continue
    img = crop(f); rects, cc = refs[pl].proposals(img)
    if cc < 0.6 or not rects:
        cands[f] = {"ref": pl, "cc": round(cc, 2), "rects": []}; continue
    cands[f] = {"ref": pl, "cc": round(cc, 2), "rects": rects[:6]}
    vis = img.copy()
    for i, (x, y, w, h) in enumerate(rects[:6], 1):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 3)
        cv2.putText(vis, str(i), (x + 6, y + 36), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 4)
    t = cv2.resize(vis, (400, 350)); cv2.putText(t, f"f{f} cc{cc:.2f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    sheets.setdefault(pl, []).append(t)
for pl, tiles in sheets.items():
    while len(tiles) % 6:
        tiles.append(np.zeros((350, 400, 3), np.uint8))
    cv2.imwrite(os.path.join(OUT, f"cands_{pl}.jpg"), np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]), [cv2.IMWRITE_JPEG_QUALITY, 75])
json.dump(cands, open(os.path.join(OUT, "candidates.json"), "w"))
for pl in REFS:
    fs = [f for f in cands if cands[f]["ref"] == pl]
    print(f"ref {pl}: {len(fs)} frames, with candidates {sum(1 for f in fs if cands[f]['rects'])}, cc median {np.median([cands[f]['cc'] for f in fs]) if fs else 0:.2f}")
reader.release()
