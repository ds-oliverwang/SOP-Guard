"""零訓練物件偵測（YOLO-World，ultralytics）＋背景執行緒包裝。

ObjectDetector：給文字描述就能找物件；一格約 0.2～0.4 秒（i5 CPU），不能每格跑。
AsyncDetector：把偵測放到背景執行緒，主迴圈 submit 最新影格、poll 拿最新結果，畫面不會被卡住。
"""
from __future__ import annotations

import os
import threading
import time

from scene import Detection


class ObjectDetector:
    def __init__(self, weights: str, prompts: list[str], imgsz: int = 640, conf: float = 0.15, threads: int | None = 2):
        if not os.path.isfile(weights):
            raise FileNotFoundError(f"找不到物件偵測權重：{os.path.abspath(weights)}")
        use_cuda = False
        try:
            import torch
            use_cuda = torch.cuda.is_available()
            if threads and not use_cuda:
                torch.set_num_threads(int(threads))   # 留 CPU 給 MediaPipe 與畫面
        except Exception:
            pass
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.prompts = list(prompts)
        self.model.set_classes(self.prompts)
        self.imgsz = imgsz
        self.conf = conf
        self.device = "0" if use_cuda else "cpu"
        self.last_ms = 0.0

    def detect(self, frame_bgr) -> list[Detection]:
        t0 = time.perf_counter()
        kw = {"device": self.device} if self.device != "cpu" else {}
        r = self.model.predict(frame_bgr, imgsz=self.imgsz, conf=self.conf, verbose=False, **kw)[0]
        out: list[Detection] = []
        for b in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            out.append(Detection(self.prompts[int(b.cls)], float(b.conf), (x1, y1, x2, y2)))
        self.last_ms = (time.perf_counter() - t0) * 1000
        return out


class AsyncDetector:
    """背景執行緒跑任何有 detect(frame) 的物件。只保留「最新一張」待處理影格，忙不過來就丟掉舊的。"""

    def __init__(self, detector):
        self._det = detector
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._pending = None          # (frame, t)
        self._result = None           # (dets, t)
        self._busy = False
        self._stop = False
        self.runs = 0
        self.total_ms = 0.0
        self.error: Exception | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame, t: float) -> bool:
        """放一張影格進去；若上一張還沒開始處理就被取代。回傳是否接受（忙碌中仍接受，會排到下一輪）。"""
        with self._cv:
            if self._stop:
                return False
            self._pending = (frame.copy(), t)
            self._cv.notify()
        return True

    def busy(self) -> bool:
        with self._lock:
            return self._busy or self._pending is not None

    def poll(self):
        """取走最新結果 (dets, t)；沒有新結果回傳 None。"""
        with self._lock:
            r = self._result
            self._result = None
            return r

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                frame, t = self._pending
                self._pending = None
                self._busy = True
            try:
                t0 = time.perf_counter()
                dets = self._det.detect(frame)
                self.total_ms += (time.perf_counter() - t0) * 1000
                self.runs += 1
                with self._lock:
                    self._result = (dets, t)
            except Exception as e:  # 背景執行緒不可無聲死掉
                self.error = e
            finally:
                with self._lock:
                    self._busy = False

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.runs if self.runs else 0.0

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(timeout=5.0)
