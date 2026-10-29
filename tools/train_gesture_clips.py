"""從 <專案>/gestures/<動作名>/*.avi|mp4 訓練手勢分類器（每段影片一個標籤，不用手動標時間）。

usage: python tools/train_gesture_clips.py --project <專案> [--yolo models/yolov8s-worldv2.pt] [--folds 5]
       [--min-clips 5] [--out gesture_model.pkl]
流程：每段影片抽手部關鍵點（逐格）＋專案 gesture_anchors 的物件框（每 15 格，快取到 gestures/_cache/）
      → 切 1 秒視窗 → 依片段分組交叉驗證 → 用全部資料訓練 → 存進專案 → 把專案 config 的 gesture_model 設好。
片段太少的動作會提醒；少於 --min-clips 的動作不訓練。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gesture_classifier import GestureClassifier, N_ANCHORS  # noqa: E402
from gesture_data import Clip, anchor_rects_per_frame, build_dataset, cross_validate_by_clip, format_report  # noqa: E402
from project_store import app_base_dir, load_project  # noqa: E402

VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv")


def extract_clip(video_path: str, hand_model: str, yolo, prompts: list[str], every: int = 15) -> dict:
    """抽一段影片的手部與物件框（與 tools/extract_features.py 相同格式，但在記憶體內）。"""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    from video_recorder import VideoFileSource
    from core_detector import HAND_DET_CONF
    opts = vision.HandLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=hand_model),
                                        running_mode=vision.RunningMode.VIDEO, num_hands=2,
                                        min_hand_detection_confidence=HAND_DET_CONF, min_tracking_confidence=HAND_DET_CONF)
    det = vision.HandLandmarker.create_from_options(opts)
    src = VideoFileSource(video_path)
    W, H = src.resolution
    hands = []; dets = []; last_ts = -1; i = -1
    scale = 640 / max(W, H); dw, dh = int(W * scale), int(H * scale)
    while True:
        frame = src.read()
        if frame is None:
            break
        i += 1
        rgb = cv2.cvtColor(cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        ts = int(i / src.fps * 1000); ts = ts if ts > last_ts else last_ts + 1; last_ts = ts
        r = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        h = np.full((2, 21, 2), np.nan, np.float32)
        for k, hand in enumerate(r.hand_landmarks[:2]):
            h[k] = [(lm.x * W, lm.y * H) for lm in hand]
        hands.append(h)
        if yolo is not None and i % every == 0:
            for d in yolo.detect(frame):
                dets.append([i, prompts.index(d.label), d.conf, *d.xyxy])
    src.release(); det.close()
    return {"hands": np.array(hands, np.float32).reshape(-1, 2, 21, 2), "det": np.array(dets, np.float32).reshape(-1, 7),
            "labels": prompts, "fps": src.fps}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--hand-model", default=os.path.join(app_base_dir(), "models", "hand_landmarker.task"))
    ap.add_argument("--yolo", default=os.path.join(app_base_dir(), "models", "yolov8s-worldv2.pt"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-clips", type=int, default=5)
    ap.add_argument("--out", default="gesture_model.pkl")
    ap.add_argument("--no-write-config", action="store_true")
    args = ap.parse_args()

    project = load_project(args.project)
    base = os.path.join(project.dir_path, "gestures")
    if not os.path.isdir(base):
        print(f"找不到片段資料夾：{base}（請先用 tools/record_clips.py 錄）"); return 1
    cache_dir = os.path.join(base, "_cache"); os.makedirs(cache_dir, exist_ok=True)
    anchor_prompts = [next(a.prompt for a in project.anchors if a.name == n) for n in project.gesture_anchors]
    if len(anchor_prompts) > N_ANCHORS:
        print(f"gesture_anchors 最多 {N_ANCHORS} 個"); return 1
    yolo = None
    if anchor_prompts:
        from object_detector import ObjectDetector
        yolo = ObjectDetector(args.yolo, anchor_prompts)

    clips: list[Clip] = []
    per_label: dict[str, int] = {}
    for label in sorted(os.listdir(base)):
        folder = os.path.join(base, label)
        if label.startswith("_") or not os.path.isdir(folder):
            continue
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(VIDEO_EXTS))
        per_label[label] = len(files)
        for f in files:
            path = os.path.join(folder, f)
            cache = os.path.join(cache_dir, f"{label}__{os.path.splitext(f)[0]}.npz")
            if os.path.isfile(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
                d = np.load(cache, allow_pickle=False)
                data = {"hands": d["hands"], "det": d["det"], "labels": [str(s) for s in d["labels"]]}
            else:
                print(f"抽特徵：{label}/{f}", flush=True)
                data = extract_clip(path, args.hand_model, yolo, anchor_prompts)
                np.savez_compressed(cache, hands=data["hands"], det=data["det"], labels=np.array(anchor_prompts))
            n = len(data["hands"])
            rects = [anchor_rects_per_frame(data["det"], data["labels"], p, n) for p in anchor_prompts]
            clips.append(Clip(label=label, name=f, hands=data["hands"], rects=rects,
                              frame_size=(project.resolution[0], project.resolution[1])))

    few = [l for l, n in per_label.items() if n < args.min_clips]
    for l in few:
        print(f"注意：「{l}」只有 {per_label[l]} 段（少於 {args.min_clips}），這一類不訓練")
    clips = [c for c in clips if c.label not in few]
    if len({c.label for c in clips}) < 2:
        print("至少要有 2 個動作類別（含「其他」）才能訓練"); return 1
    X, y, groups = build_dataset(clips)
    print(f"視窗 {len(y)} 個、特徵 {X.shape[1]} 維")
    print(format_report(cross_validate_by_clip(X, y, groups, n_folds=args.folds)))
    clf = GestureClassifier(); clf.fit(X, y)
    out = os.path.join(project.dir_path, args.out)
    clf.save(out)
    print(f"已用全部片段訓練並存檔：{out}")
    if not args.no_write_config:
        cfg_path = os.path.join(project.dir_path, "config.json")
        with open(cfg_path, "r", encoding="utf-8-sig") as f:
            cfg = json.load(f)
        cfg["gesture_model"] = args.out
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        print(f"已把 {cfg_path} 的 gesture_model 設為 {args.out}；步驟請用 type=\"gesture\"、label=動作名")
    return 0


if __name__ == "__main__":
    sys.exit(main())
