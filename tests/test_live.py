"""背景執行緒偵測與錄影元件（不需要相機、不需要 YOLO）。"""
from __future__ import annotations

import os
import time

import numpy as np

from object_detector import AsyncDetector
from scene import Detection
from video_recorder import Recorder, is_camera_source, is_network_source, open_video_oriented


def test_source_classification():
    assert is_camera_source(0) and is_camera_source("0") and is_camera_source(" 1 ")
    assert is_camera_source("rtsp://admin:pw@192.168.1.10:554/stream1") and is_network_source("RTSP://x/y")
    assert is_camera_source("http://192.168.1.10/video.mjpg")
    assert not is_camera_source(r"C:\videos\demo.mp4") and not is_camera_source("demo.avi")
    assert not is_network_source(0)


class FakeDetector:
    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        time.sleep(self.delay)
        return [Detection("sneaker", 0.9, (0, 0, float(frame.shape[1]), float(frame.shape[0])))]


def test_async_detector_returns_latest_and_drops_backlog():
    fake = FakeDetector(delay=0.05)
    det = AsyncDetector(fake)
    frame = np.zeros((10, 20, 3), np.uint8)
    assert det.poll() is None
    det.submit(frame, 1.0)
    deadline = time.time() + 2.0
    while time.time() < deadline and fake.calls < 1:   # 等背景執行緒真的開始處理第一張
        time.sleep(0.002)
    for i in range(5):                    # 處理中連續丟 5 張，只會留最後一張
        det.submit(frame, 2.0 + i)
    deadline = time.time() + 2.0
    results = []
    while time.time() < deadline and len(results) < 2:
        r = det.poll()
        if r is not None:
            results.append(r)
        time.sleep(0.01)
    det.close()
    assert len(results) == 2
    assert results[0][1] == 1.0 and results[1][1] == 6.0
    assert results[1][0][0].label == "sneaker" and results[1][0][0].rect == (0, 0, 20, 10)
    assert fake.calls == 2


def test_async_detector_surfaces_errors():
    class Boom:
        def detect(self, frame):
            raise RuntimeError("模型壞了")
    det = AsyncDetector(Boom())
    det.submit(np.zeros((4, 4, 3), np.uint8), 0.0)
    deadline = time.time() + 2.0
    while time.time() < deadline and det.error is None:
        time.sleep(0.01)
    det.close()
    assert isinstance(det.error, RuntimeError)


def test_recorder_writes_mjpg_avi(tmp_path):
    path = str(tmp_path / "rec" / "demo.avi")
    rec = Recorder()
    rec.start(path, (64, 48), 30.0)
    for i in range(10):
        rec.write(np.full((48, 64, 3), i * 20, np.uint8))
    rec.stop()
    assert rec.frame_count == 10 and os.path.getsize(path) > 0
    cap, rotate, w, h, fps = open_video_oriented(path)
    cap.release()
    assert (w, h) == (64, 48) and rotate is None and abs(fps - 30.0) < 0.1
