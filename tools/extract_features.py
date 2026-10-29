"""把一支影片的「手部關鍵點（每格）」與「物件偵測框（每 N 格）」抽成快取檔，後續實驗直接讀快取。

usage: python tools/extract_features.py <video> <out.npz> [--hand-model models/hand_landmarker.task]
       [--yolo <weights.pt>] [--prompts "cardboard box,sneaker"] [--every 15] [--imgsz 640] [--conf 0.15] [--max-frames N]

npz 內容：
  hands   (F, 2, 21, 2) float32  每格最多兩隻手的 21 點像素座標（轉正後），沒有的手填 NaN
  handed  (F, 2)        int8     0=Left 1=Right -1=無
  det     (M, 7)        float32  [frame_idx, label_idx, conf, x1, y1, x2, y2]
  labels  (L,)          str      prompts 順序
  meta    [fps, W, H, every]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from video_recorder import VideoFileSource  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--hand-model", default=os.path.join(ROOT, "models", "hand_landmarker.task"))
    ap.add_argument("--yolo", default=None, help="YOLO-World 權重；不給就不做物件偵測")
    ap.add_argument("--prompts", default="cardboard box,sneaker")
    ap.add_argument("--every", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    from core_detector import HAND_DET_CONF
    hand_opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=args.hand_model),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=HAND_DET_CONF, min_tracking_confidence=HAND_DET_CONF)
    hand_det = vision.HandLandmarker.create_from_options(hand_opts)

    yolo = None
    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    if args.yolo:
        from ultralytics import YOLO
        yolo = YOLO(args.yolo)
        yolo.set_classes(prompts)

    src = VideoFileSource(args.video)
    W, H = src.resolution
    fps = src.fps
    total = src.frame_count if not args.max_frames else min(src.frame_count, args.max_frames)
    hands = np.full((total, 2, 21, 2), np.nan, np.float32)
    handed = np.full((total, 2), -1, np.int8)
    dets: list[list[float]] = []
    scale = 640 / max(W, H)
    dw, dh = int(W * scale), int(H * scale)
    t0 = time.perf_counter(); last_ts = -1; hand_ms = 0.0; yolo_ms = 0.0; n_yolo = 0
    i = -1
    while True:
        frame = src.read()
        if frame is None:
            break
        i += 1
        if i >= total:
            break
        t_sec = i / fps
        small = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        ts = int(t_sec * 1000); ts = ts if ts > last_ts else last_ts + 1; last_ts = ts
        h0 = time.perf_counter()
        res = hand_det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        hand_ms += (time.perf_counter() - h0) * 1000
        for k, hand in enumerate(res.hand_landmarks[:2]):
            hands[i, k] = [(lm.x * W, lm.y * H) for lm in hand]
            name = res.handedness[k][0].category_name if res.handedness and len(res.handedness) > k else ""
            handed[i, k] = 0 if name == "Left" else 1 if name == "Right" else -1
        if yolo is not None and i % args.every == 0:
            y0 = time.perf_counter()
            r = yolo.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            yolo_ms += (time.perf_counter() - y0) * 1000; n_yolo += 1
            for b in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                dets.append([i, int(b.cls), float(b.conf), x1, y1, x2, y2])
        if i % 300 == 0:
            el = time.perf_counter() - t0
            print(f"frame {i}/{total}  {el:.0f}s elapsed  hand avg {hand_ms / (i + 1):.1f} ms  yolo avg {yolo_ms / max(n_yolo, 1):.0f} ms", flush=True)
    src.release(); hand_det.close()
    n = i + 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, hands=hands[:n], handed=handed[:n],
                        det=np.array(dets, np.float32).reshape(-1, 7), labels=np.array(prompts),
                        meta=np.array([fps, W, H, args.every], np.float32))
    with_hand = int((~np.isnan(hands[:n, 0, 0, 0])).sum())
    print(f"done frames={n} hand_frames={with_hand} ({with_hand / n * 100:.1f}%) detections={len(dets)} "
          f"hand_avg_ms={hand_ms / n:.1f} yolo_avg_ms={yolo_ms / max(n_yolo, 1):.0f} wall={time.perf_counter() - t0:.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
