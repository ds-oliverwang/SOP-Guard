"""Contact sheet of the example crops the simulator picked (cache/_sim_lib2/states), grouped by state."""
import os, sys, json
import numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "cache", "_sim_lib2", "states")
d = json.load(open(os.path.join(LIB, "states.json"), encoding="utf-8"))
tw, th = 240, 170
rows = []
for st in d["order"]:
    tiles = []
    for e in d["examples"]:
        if e["state"] != st:
            continue
        img = cv2.imdecode(np.fromfile(os.path.join(LIB, e["file"]), np.uint8), cv2.IMREAD_COLOR)
        t = cv2.resize(img, (tw, th)); cv2.putText(t, f"{st} {e['frame']}", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        tiles.append(t)
    while len(tiles) < 7:
        tiles.append(np.zeros((th, tw, 3), np.uint8))
    rows.append(np.hstack(tiles[:7]))
out = os.path.join(ROOT, "cache", "state_windows", "examples_inner.jpg")
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85]); print("wrote", out)
