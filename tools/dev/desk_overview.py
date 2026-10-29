"""Overview of the user's new desk recording: frame count, contact sheet of sampled frames with YOLO 'cardboard box'
detections drawn, and the frames around the segments they labelled."""
import os, sys
import numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import torch  # noqa
from video_recorder import FrameReader
from object_detector import ObjectDetector
P = os.path.join(ROOT, "sop_projects", "桌面狀態測試", "demo.avi")
OUT = os.path.join(ROOT, "cache", "desk_overview")
os.makedirs(OUT, exist_ok=True)
reader = FrameReader(P)
N = reader.frame_count
print("frames", N, "sec", N / 30)
det = ObjectDetector(os.path.join(ROOT, "models", "yolov8s-worldv2.pt"), ["cardboard box", "box", "sneaker"], conf=0.2, threads=2)
def sheet(name, frames, cols=6, tw=400, th=225):
    tiles = []
    for f in frames:
        f = min(max(0, f), N - 1)
        fr = reader.get_frame(f)
        if fr is None:
            continue
        vis = fr.copy()
        dets = det.detect(fr)
        summary = []
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d.xyxy]
            col = (0, 255, 255) if d.label == "cardboard box" else (255, 200, 0) if d.label == "box" else (0, 255, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 6)
            cv2.putText(vis, f"{d.label} {d.conf:.2f}", (x1 + 6, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
            summary.append(f"{d.label}:{d.conf:.2f}@{x1},{y1},{x2 - x1}x{y2 - y1}")
        print(f"f={f:5d} t={f / 30:6.1f}s  {'; '.join(summary) or '(no detection)'}")
        t = cv2.resize(vis, (tw, th)); cv2.putText(t, f"{f} ({f / 30:.0f}s)", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.zeros((th, tw, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    img = np.vstack(rows); cv2.imwrite(os.path.join(OUT, name), img, [cv2.IMWRITE_JPEG_QUALITY, 80]); print("wrote", name, img.shape)
print("--- overview every 90 frames")
sheet("overview.jpg", list(range(0, N, 90)))
print("--- around labelled segments 放鞋盒 534-700 / 放標籤 730-1000 / 放鞋子 1012-1165")
sheet("round1.jpg", [500, 534, 600, 700, 715, 730, 800, 900, 1000, 1006, 1012, 1100, 1165, 1200, 1260, 1320, 1380, 1440])
reader.release()
