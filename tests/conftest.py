"""測試共用：在 tmp_path 建立規格 3.3 的兩步驟範例專案，並提供合成手部座標的餵入工具。"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import core_detector  # noqa: E402
from project_store import load_project  # noqa: E402
from video_recorder import write_image  # noqa: E402

FPS = 30.0
FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)

ROI0 = [100, 200, 150, 150]
ROI1 = [450, 300, 200, 180]
RESET_ZONE = [1100, 600, 150, 100]
IN_ROI0 = [(175.0, 275.0)]
IN_ROI1 = [(550.0, 390.0)]
IN_RESET = [(1175.0, 650.0)]
NO_HAND: list = []


def sample_config(**overrides) -> dict:
    cfg = {
        "schema_version": 1,
        "project_name": "範例_貼標與包裝作業",
        "video_file": "demo.avi",
        "resolution": [1280, 720],
        "fps": FPS,
        "skip_alert_sec": 1.0,
        "cycle_arm_clear_sec": 0.5,
        "reset_zone": None,
        "reset_dwell_sec": 1.0,
        "steps": [
            {"step_index": 0, "step_name": "拿取標籤", "frame_number": 42, "roi": list(ROI0),
             "trigger_mode": "hand_only", "dwell_time_sec": 0.4, "timeout_sec": 10.0,
             "visual_match_min": None, "template_file": "templates/step_0_roi.jpg"},
            {"step_index": 1, "step_name": "貼合於產品表面", "frame_number": 115, "roi": list(ROI1),
             "trigger_mode": "hand_and_visual", "dwell_time_sec": 0.5, "timeout_sec": 8.0,
             "visual_match_min": 0.75, "template_file": "templates/step_1_roi.jpg"},
        ],
    }
    cfg.update(overrides)
    return cfg


def write_project(dir_path: str, cfg: dict) -> str:
    os.makedirs(os.path.join(dir_path, "templates"), exist_ok=True)
    with open(os.path.join(dir_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    for s in cfg["steps"]:
        write_image(os.path.join(dir_path, s["template_file"]), np.full((40, 40, 3), 128, dtype=np.uint8))
    return dir_path


@pytest.fixture
def project_dir(tmp_path):
    return write_project(str(tmp_path / "proj"), sample_config())


@pytest.fixture
def project(project_dir):
    return load_project(project_dir)


class Clock:
    def __init__(self):
        self.t = 0.0


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def make_fsm():
    def _make(project, **kw):
        kw.setdefault("error_log_dir", None)
        kw.setdefault("sound_enabled", False)
        return core_detector.SOPStateMachine(project, **kw)
    return _make


def feed(fsm, points, seconds: float, clock: Clock, fps: float = FPS) -> None:
    """以 1/fps 的固定 dt 連續餵 seconds 秒的影格。第一次呼叫的第一格 dt = 0。"""
    n = int(round(seconds * fps))
    for _ in range(n):
        fsm.update(FRAME, points, clock.t)
        clock.t += 1.0 / fps
