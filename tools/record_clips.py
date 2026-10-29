"""錄手勢示範片段：每段一個動作，存到 <專案>/gestures/<動作名>/<動作名>_NNN.avi（MJPG，相機實際 fps）。

usage: python tools/record_clips.py --project <專案> --labels "確認尺碼,放鞋墊,其他" [--camera 0]
視窗操作：空白鍵 開始／停止錄一段；n 切到下一個動作；p 上一個；q 離開。
畫面會顯示目前動作、已錄幾段、正在錄的秒數；建議每個動作錄 15～30 段、每段 2～5 秒；
「其他」這一類請錄各種不是目標動作的手部動作（空手移動、整理桌面…），分類器才知道什麼叫「不是」。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core_detector import draw_texts  # noqa: E402
from project_store import load_project  # noqa: E402
from video_recorder import CameraSource, Recorder  # noqa: E402


def next_index(folder: str, label: str) -> int:
    os.makedirs(folder, exist_ok=True)
    n = 0
    for f in os.listdir(folder):
        if f.startswith(label + "_") and f.lower().endswith(".avi"):
            try:
                n = max(n, int(f[len(label) + 1:-4]))
            except ValueError:
                pass
    return n + 1


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()
    project = load_project(args.project)
    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    base = os.path.join(project.dir_path, "gestures")
    cam = CameraSource(args.camera, project.resolution[0], project.resolution[1])
    print(f"相機 {args.camera}：{cam.resolution[0]}x{cam.resolution[1]} @ {cam.fps:.1f} fps；片段存到 {base}")
    rec = Recorder(); cur = 0; rec_start = 0.0
    counts = {l: next_index(os.path.join(base, l), l) - 1 for l in labels}
    try:
        while True:
            frame = cam.read()
            if frame is None:
                break
            if rec.recording:
                rec.write(frame)
            vis = cv2.resize(frame, (960, int(960 * frame.shape[0] / frame.shape[1]))) if frame.shape[1] > 960 else frame.copy()
            texts = [(10, 6, f"動作 {cur + 1}/{len(labels)}：{labels[cur]}   已錄 {counts[labels[cur]]} 段", (255, 255, 255), 26)]
            if rec.recording:
                texts.append((10, 40, f"● 錄製中 {time.perf_counter() - rec_start:.1f} 秒（空白鍵停止）", (0, 0, 255), 24))
                cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), (0, 0, 255), 8)
            else:
                texts.append((10, 40, "空白鍵開始錄一段；n 下一個動作；p 上一個；q 離開", (0, 200, 255), 20))
            cv2.imshow("錄手勢片段", draw_texts(vis, texts))
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord(" "):
                if rec.recording:
                    rec.stop(); counts[labels[cur]] += 1
                    print(f"已存 {rec.path}（{rec.frame_count} 格）")
                else:
                    lab = labels[cur]; folder = os.path.join(base, lab)
                    rec.start(os.path.join(folder, f"{lab}_{next_index(folder, lab):03d}.avi"), cam.resolution, cam.fps)
                    rec_start = time.perf_counter()
            elif k == ord("n") and not rec.recording:
                cur = (cur + 1) % len(labels)
            elif k == ord("p") and not rec.recording:
                cur = (cur - 1) % len(labels)
    finally:
        if rec.recording:
            rec.stop()
        cam.release(); cv2.destroyAllWindows()
    print("各動作片段數：" + "、".join(f"{l} {counts[l]}" for l in labels))
    return 0


if __name__ == "__main__":
    sys.exit(main())
