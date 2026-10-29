"""錨點探測：只跑 YOLO-World 並把偵測框畫出來，用來在桌面上調 prompt 與信心門檻。

usage: python tools/anchor_probe.py --source 0|<影片> --prompts "cardboard box,sneaker" [--conf 0.15] [--imgsz 640]
       [--yolo models/yolov8s-worldv2.pt] [--max-seconds 0] [--save <jpg資料夾>]
視窗裡：每個框標「類別 信心 面積」；左上角列每個類別最近 5 秒出現的比例與最高信心。按 q 離開。
結束時印每個類別的出現比例，方便決定 min_conf / min_area。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core_detector import draw_texts  # noqa: E402
from object_detector import AsyncDetector, ObjectDetector  # noqa: E402
from project_store import app_base_dir  # noqa: E402
from video_recorder import CameraSource, VideoFileSource, is_camera_source, write_image  # noqa: E402


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--yolo", default=os.path.join(app_base_dir(), "models", "yolov8s-worldv2.pt"))
    ap.add_argument("--max-seconds", type=float, default=0)
    ap.add_argument("--save", default=None, help="每 5 秒存一張畫面到此資料夾")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    det = AsyncDetector(ObjectDetector(args.yolo, prompts, imgsz=args.imgsz, conf=args.conf))
    src = CameraSource(args.source.strip()) if is_camera_source(args.source) else VideoFileSource(args.source)
    print(f"來源 {src.resolution[0]}x{src.resolution[1]} @ {src.fps:.1f} fps；prompts={prompts} conf≥{args.conf}")
    colors = [(0, 200, 0), (255, 120, 0), (0, 0, 255), (255, 0, 255), (0, 200, 255), (200, 200, 0)]
    latest = []            # 最新偵測結果
    seen = defaultdict(int); runs = 0; best = defaultdict(float)
    recent = defaultdict(list)   # 類別 → 最近出現時間
    t0 = time.perf_counter(); next_save = 5.0
    try:
        while True:
            frame = src.read()
            if frame is None:
                break
            t = src.time_sec
            if not det.busy():
                det.submit(frame, t)
            r = det.poll()
            if r is not None:
                latest = r[0]; runs += 1
                present = {d.label for d in latest}
                for p in present:
                    seen[p] += 1; recent[p].append(t)
                for d in latest:
                    best[d.label] = max(best[d.label], d.conf)
            if det.error is not None:
                raise det.error
            scale = 960 / max(frame.shape[:2])
            vis = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale))) if scale < 1 else frame.copy()
            texts = []
            for d in latest:
                x1, y1, x2, y2 = d.xyxy
                c = colors[prompts.index(d.label) % len(colors)]
                cv2.rectangle(vis, (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale)), c, 2)
                texts.append((int(x1 * scale) + 3, int(y1 * scale) + 2, f"{d.label} {d.conf:.2f} 面積{int(d.area / 1000)}k", c, 18))
            y = 6
            for i, p in enumerate(prompts):
                rec = [x for x in recent[p] if t - x <= 5.0]
                ratio = len(rec) / max(1, len([1 for _ in range(1)]) * max(1, runs_in_window(runs, det)))
                texts.append((8, y, f"{p}: 近5秒出現 {len(rec)} 次  最高信心 {best[p]:.2f}", colors[i % len(colors)], 20)); y += 26
            texts.append((8, y, f"偵測 {det.avg_ms:.0f} ms/格  已跑 {runs} 次  t={t:.0f}s", (255, 255, 255), 18))
            vis = draw_texts(vis, texts)
            if args.save and t >= next_save:
                next_save += 5.0
                os.makedirs(args.save, exist_ok=True)
                write_image(os.path.join(args.save, f"probe_{int(t):04d}s.jpg"), vis)
            if not args.no_show:
                cv2.imshow("anchor probe（按 q 離開）", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_seconds and (time.perf_counter() - t0) >= args.max_seconds:
                break
    finally:
        det.close(); src.release()
        if not args.no_show:
            cv2.destroyAllWindows()
    print(f"偵測共 {runs} 次，平均 {det.avg_ms:.0f} ms/格")
    for p in prompts:
        print(f"  {p:16s} 出現於 {seen[p]}/{runs} 次偵測（{seen[p] / max(runs, 1) * 100:.0f}%），最高信心 {best[p]:.2f}")
    return 0


def runs_in_window(runs: int, det) -> int:
    """近 5 秒大約跑了幾次偵測（用平均耗時估）。"""
    return max(1, int(5000 / det.avg_ms)) if det.avg_ms else 1


if __name__ == "__main__":
    sys.exit(main())
