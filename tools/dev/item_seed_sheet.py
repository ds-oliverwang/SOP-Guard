"""Render box crops of chosen frames with a 100-px coordinate grid so seed rectangles can be read off by eye."""
import os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
from item_core import *   # noqa
FR = [1000, 1100, 2000, 2120, 2850, 2960, 3700, 3850, 4700, 4865]
tiles = []
for f in FR:
    f5 = f - f % STEP
    reg = status[f5][0]
    if reg is None:
        # nearest frame with a region
        for d in range(5, 200, 5):
            if status.get(f5 - d, (None,))[0] is not None:
                reg = status[f5 - d][0]; break
    img = crop_roi(reader.get_frame(f), reg)
    h, w = img.shape[:2]
    for x in range(0, w, 100):
        cv2.line(img, (x, 0), (x, h), (0, 255, 255) if x % 500 == 0 else (0, 160, 160), 2 if x % 500 == 0 else 1)
        cv2.putText(img, str(x), (x + 2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    for y in range(0, h, 100):
        cv2.line(img, (0, y), (w, y), (0, 255, 255) if y % 500 == 0 else (0, 160, 160), 2 if y % 500 == 0 else 1)
        cv2.putText(img, str(y), (2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(img, f"f{f} crop {w}x{h} free={status[f5][1]}", (120, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    tiles.append(cv2.resize(img, (560, 500)))
rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
cv2.imwrite(os.path.join(OUT, "seed_grid.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80]); print("ok")
reader.release()
