"""用快取（tools/extract_features.py 產出的 npz）離線跑 v2 狀態機，不需要重新偵測。

usage: python tools/run_cached.py <features.npz> <project_dir> [--auto-reset-sec 2.0] [--max-frames N] [--print-all]
輸出與 core_detector.py 相同格式的事件與摘要（沒有 FPS，因為不做偵測）。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core_detector import EVENT_TYPES, HAND_POINT_IDS, FrameContext, SOPStateMachine, union_rect  # noqa: E402
from project_store import load_project  # noqa: E402
from scene import Detection, SceneTracker  # noqa: E402


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("npz"); ap.add_argument("project")
    ap.add_argument("--auto-reset-sec", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--print-all", action="store_true", help="連 step_done / armed 也印")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=False)
    hands, det, labels = d["hands"], d["det"], [str(s) for s in d["labels"]]
    # tools/extract_features.py 的快取有 meta；training.py 的快取（gestures/_cache/demo.npz）只有 fps
    fps = float(d["meta"][0]) if "meta" in d.files else float(d["fps"])
    n = len(hands) if not args.max_frames else min(len(hands), args.max_frames)
    by_frame: dict[int, list[Detection]] = {}
    for row in det:
        f, li, c, x1, y1, x2, y2 = row
        by_frame.setdefault(int(f), []).append(Detection(labels[int(li)], float(c), (float(x1), float(y1), float(x2), float(y2))))

    project = load_project(args.project)
    for a in project.anchors:
        if a.prompt not in labels:
            print(f"注意：錨點「{a.name}」的 prompt「{a.prompt}」不在快取的偵測類別 {labels} 裡，永遠找不到")
    scene = SceneTracker(project.anchors, tuple(project.work_area) if project.work_area else None)
    fsm = SOPStateMachine(project, error_log_dir=None, sound_enabled=False)
    gesture = None
    if project.gesture_model:
        from gesture_classifier import GestureClassifier, GestureStream, hand_frame_features
        gesture = GestureStream(GestureClassifier.load(os.path.join(project.dir_path, project.gesture_model)))

    printed = 0
    for i in range(n):
        t = i / fps
        if i in by_frame:
            scene.update_detections(by_frame[i], t)
        anchors = scene.anchors_at(t)
        pts = []
        for k in range(2):
            if not np.isnan(hands[i, k, 0, 0]):
                pts.extend((float(hands[i, k, j, 0]), float(hands[i, k, j, 1])) for j in HAND_POINT_IDS)
        probs = {}
        if gesture is not None:
            refs = [union_rect(anchors.get(name, [])) for name in project.gesture_anchors]
            probs = dict(gesture.push(hand_frame_features(hands[i], refs, (project.resolution[0], project.resolution[1]))))
        fsm.update(None, FrameContext(hand_points=pts, hands_xy=hands[i], anchors=anchors, gesture_probs=probs), t)
        if fsm.alert_active and (t - fsm.alert_started_at) >= args.auto_reset_sec:
            fsm.reset()
        while printed < len(fsm.events):
            ev = fsm.events[printed]; printed += 1
            if args.print_all or ev["event_type"] in ("cycle_done", "alert_skip", "alert_timeout", "alert_missing"):
                idx = f" {ev['step_index']}" if ev["step_index"] != "" else ""
                print(f"[t={ev['t']:.2f}s] {ev['event_type']}{idx} {ev['message']}".rstrip())
    counts = {k: 0 for k in EVENT_TYPES}
    for ev in fsm.events:
        counts[ev["event_type"]] += 1
    print(f"cycles={fsm.cycle_count} bad_cycles={fsm.bad_cycle_count} discarded={counts['cycle_discarded']} step_done={counts['step_done']} "
          f"alert_skip={counts['alert_skip']} alert_missing={counts['alert_missing']} alert_timeout={counts['alert_timeout']} frames={n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
