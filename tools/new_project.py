"""建立 schema 2 專案骨架（清單模式），解析度可以直接從相機讀。

usage: python tools/new_project.py <專案名稱> (--camera 0 | --resolution 1280x720) [--fps 30]
       [--anchors "box=cardboard box,item=sneaker"] [--work-area x,y,w,h] [--ordered]

產出 sop_projects/<名稱>/config.json：
  - 每個錨點一個同名 zone（<名稱>_zone），第一個錨點另有內側區 <名稱>_inner（去掉四邊各 10%）
  - 步驟樣板：手碰第二個錨點 → 手伸進第一個錨點內側（1 秒）→（選配）第二個錨點的物件落在第一個錨點裡
  - cycle_end：第二個錨點（物品）離開畫面 2 秒；cycle_min_sec 10
建好後請用文字編輯器改 step_name、dwell_time_sec 等。
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from project_store import ProjectConfigError, load_project, projects_dir, validate_project_name  # noqa: E402
import json  # noqa: E402


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--resolution", default=None, help="例如 1280x720")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--anchors", default="box=cardboard box,item=sneaker", help="名稱=prompt，逗號分隔；第一個是容器、第二個是物品")
    ap.add_argument("--work-area", default=None, help="x,y,w,h；不給就整個畫面")
    ap.add_argument("--ordered", action="store_true", help="用依序模式（預設清單模式）")
    ap.add_argument("--force", action="store_true", help="已存在就覆蓋 config.json")
    args = ap.parse_args()

    validate_project_name(args.name)
    if args.camera is not None:
        from video_recorder import CameraSource
        cam = CameraSource(args.camera)
        w, h = cam.resolution; fps = args.fps or cam.fps
        cam.release()
        print(f"相機 {args.camera}：{w}x{h} @ {fps:.1f} fps")
    elif args.resolution:
        w, h = [int(v) for v in args.resolution.lower().split("x")]; fps = args.fps or 30.0
    else:
        print("要給 --camera 或 --resolution"); return 2

    pairs = [p.split("=", 1) for p in args.anchors.split(",") if "=" in p]
    if not pairs:
        print("--anchors 格式：名稱=prompt,名稱=prompt"); return 2
    container, item = pairs[0][0].strip(), (pairs[1][0].strip() if len(pairs) > 1 else None)
    anchors = []
    for i, (n, p) in enumerate(pairs):
        a = {"name": n.strip(), "prompt": p.strip(), "min_conf": 0.2, "pick": "largest" if i == 0 else "all",
             "max_count": 1 if i == 0 else 2, "min_area": 0, "max_age_sec": 2.0}
        if i == 0 and item:
            a["exclude_containing"] = [item]
        anchors.append(a)
    zones = [{"name": f"{a['name']}_zone", "anchor": a["name"]} for a in anchors]
    zones.append({"name": f"{container}_inner", "anchor": container, "rel": [0.1, 0.1, 0.8, 0.8]})
    steps = []
    if item:
        steps.append({"step_index": 0, "step_name": f"處理{item}", "type": "hand_dwell", "zone": f"{item}_zone",
                      "dwell_time_sec": 0.4, "timeout_sec": 30.0})
    steps.append({"step_index": len(steps), "step_name": f"手伸進{container}", "type": "hand_dwell", "zone": f"{container}_inner",
                  "dwell_time_sec": 1.0, "timeout_sec": 30.0})
    if item:
        steps.append({"step_index": len(steps), "step_name": f"{item}放入{container}", "type": "object_in_zone", "object": item,
                      "zone": f"{container}_zone", "min_overlap": 0.5, "count": "any", "dwell_time_sec": 1.0,
                      "timeout_sec": 30.0, "required": False})
    cfg = {
        "schema_version": 2, "project_name": args.name, "video_file": "demo.avi", "resolution": [w, h], "fps": round(fps, 2),
        "skip_alert_sec": 1.0, "cycle_arm_clear_sec": 0.5, "reset_zone": None, "reset_dwell_sec": 1.0,
        "work_area": [int(v) for v in args.work_area.split(",")] if args.work_area else None,
        "detect_every_n_frames": 15, "gesture_model": None, "gesture_anchors": [],
        "cycle_end": None if args.ordered else {"type": "object_absent", "object": item or container, "hold_sec": 2.0},
        "cycle_timeout_sec": None if args.ordered else 120.0, "cycle_min_sec": 0.0 if args.ordered else 10.0,
        "anchors": anchors, "zones": zones, "steps": steps,
    }
    d = os.path.join(projects_dir(), args.name)
    path = os.path.join(d, "config.json")
    if os.path.exists(path) and not args.force:
        print(f"已存在：{path}（要覆蓋加 --force）"); return 1
    os.makedirs(os.path.join(d, "templates"), exist_ok=True)
    os.makedirs(os.path.join(d, "error_logs"), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    try:
        load_project(d)
    except ProjectConfigError as e:
        print(f"寫出的設定沒通過驗證：{e}"); return 1
    print(f"已建立 {path}")
    print("下一步：python core_detector.py --project sop_projects\\" + args.name + f" --source {args.camera if args.camera is not None else 0} --show")
    return 0


if __name__ == "__main__":
    sys.exit(main())
