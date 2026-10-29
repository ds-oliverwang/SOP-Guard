"""盒內狀態辨識：看「容器裡現在有什麼」，不看手怎麼動。

做法（免訓練）：
  - 操作者在示範影片上點幾格，說「這一格盒子是『空盒』／『已放卡片』／『已放鞋子』」，程式把那幾格的盒內裁圖存成範例。
  - 監控時每隔幾格裁一次盒內影像，算描述向量，跟範例做最近鄰比對 → 狀態 + 信心（1 − 距離，並看與第二名的差距）。
  - 描述向量優先用 CLIP ViT-B/32（CPU 一張約 60 ms）；環境沒有 clip 就退回 32×32 灰階縮圖（實測 82% vs CLIP 88%）。
  - 手在盒內時不判（會遮住內容），並用最近 K 次投票，避免單格閃爍。
範例存放：<專案>/states/<狀態名>/NNN.jpg ＋ <專案>/states/states.json（{"order": [...], "examples": [{"state","file","frame"}]}）。
"""
from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

STATES_DIR = "states"
STATES_FILE = "states.json"
THUMB = 32


@dataclass
class StateResult:
    state: str | None          # None = 不確定／沒有範例／手在盒內
    confidence: float = 0.0
    margin: float = 0.0        # 與第二名的距離差，越大越肯定
    reason: str = ""


class Embedder:
    """把裁圖變成向量。CLIP 可用就用 CLIP（L2 正規化），否則灰階縮圖（零均值單位變異）。"""

    def __init__(self, prefer_clip: bool = True):
        self.kind = "thumb"
        self._model = None
        if prefer_clip:
            try:
                import clip  # ultralytics 的 CLIP 分支
                import torch
                self._torch = torch
                self._model, self._preprocess = clip.load("ViT-B/32", device="cpu")
                self.kind = "clip"
            except Exception:
                self._model = None

    def embed(self, crop_bgr) -> np.ndarray:
        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError("空的裁圖")
        if self.kind == "clip":
            from PIL import Image
            img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            with self._torch.no_grad():
                e = self._model.encode_image(self._preprocess(img).unsqueeze(0)).squeeze(0).numpy().astype(np.float32)
            return e / (np.linalg.norm(e) + 1e-8)
        # 灰階縮圖：只減平均亮度、不除以標準差（除了會把「均勻的空盒」的雜訊放大成假紋理）
        g = cv2.resize(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY), (THUMB, THUMB), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        return (g - g.mean()) / 64.0 / np.sqrt(g.size)


class StateLibrary:
    """狀態範例庫：讀寫 states.json 與裁圖，並持有各範例的向量。"""

    def __init__(self, project_dir: str, embedder: Embedder | None = None):
        self.dir = os.path.join(project_dir, STATES_DIR)
        self.path = os.path.join(self.dir, STATES_FILE)
        self.embedder = embedder
        self.order: list[str] = []
        self.examples: list[dict] = []          # {"state","file","frame"}
        self._vecs: list[np.ndarray] = []
        self._labels: list[str] = []
        self.load()

    # ---- 檔案

    def load(self) -> None:
        self.order, self.examples = [], []
        if os.path.isfile(self.path):
            with open(self.path, "r", encoding="utf-8-sig") as f:
                d = json.load(f)
            self.order = [s for s in d.get("order", []) if isinstance(s, str)]
            self.examples = [e for e in d.get("examples", []) if isinstance(e, dict)]
        self._vecs, self._labels = [], []

    def save(self) -> None:
        os.makedirs(self.dir, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"order": self.order, "examples": self.examples}, f, ensure_ascii=False, indent=2)

    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for e in self.examples:
            c[e["state"]] = c.get(e["state"], 0) + 1
        return c

    def add_example(self, state: str, crop_bgr, frame: int) -> str:
        state = state.strip()
        if state not in self.order:
            self.order.append(state)
        folder = os.path.join(self.dir, state)
        os.makedirs(folder, exist_ok=True)
        n = 1 + max([int(os.path.splitext(f)[0]) for f in os.listdir(folder) if f[:-4].isdigit()] or [0])
        rel = os.path.join(state, f"{n:03d}.jpg")
        ok, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        buf.tofile(os.path.join(self.dir, rel))
        self.examples.append({"state": state, "file": rel, "frame": int(frame)})
        self._vecs, self._labels = [], []        # 下次比對重算
        self.save()
        return rel

    def remove_state(self, state: str) -> None:
        self.examples = [e for e in self.examples if e["state"] != state]
        self.order = [s for s in self.order if s != state]
        self._vecs, self._labels = [], []
        self.save()

    def remove_example(self, index: int) -> None:
        if 0 <= index < len(self.examples):
            e = self.examples.pop(index)
            try:
                os.remove(os.path.join(self.dir, e["file"]))
            except OSError:
                pass
            if not any(x["state"] == e["state"] for x in self.examples):
                self.order = [s for s in self.order if s != e["state"]]
            self._vecs, self._labels = [], []
            self.save()

    # ---- 比對

    def _ensure_vecs(self) -> None:
        if self._vecs or not self.examples:
            return
        if self.embedder is None:
            self.embedder = Embedder()
        for e in self.examples:
            p = os.path.join(self.dir, e["file"])
            img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR) if os.path.isfile(p) else None
            if img is None:
                continue
            self._vecs.append(self.embedder.embed(img)); self._labels.append(e["state"])

    def recognize(self, crop_bgr) -> StateResult:
        self._ensure_vecs()
        if not self._vecs:
            return StateResult(None, reason="沒有狀態範例")
        v = self.embedder.embed(crop_bgr)
        X = np.array(self._vecs)
        d = np.linalg.norm(X - v, axis=1)
        best: dict[str, float] = {}
        for lab, dist in zip(self._labels, d):
            best[lab] = min(best.get(lab, 9.0), float(dist))
        ranked = sorted(best.items(), key=lambda kv: kv[1])
        state, d1 = ranked[0]
        d2 = ranked[1][1] if len(ranked) > 1 else d1 + 1.0
        if self.embedder.kind == "clip":
            conf = max(0.0, 1.0 - d1 / 2.0)     # CLIP 正規化向量距離 0～2
        else:
            conf = 1.0 / (1.0 + 4.0 * d1)        # 縮圖距離：0 → 1.0，0.25 → 0.5
        return StateResult(state, conf, d2 - d1)

    def leave_one_out(self) -> dict:
        """用範例自己互驗：每張拿掉後看最近鄰是否同狀態。給介面顯示「這些範例分不分得開」。"""
        self._ensure_vecs()
        n = len(self._vecs)
        if n < 2:
            return {"n": n, "accuracy": 0.0, "confusions": {}}
        X = np.array(self._vecs); correct = 0; conf: dict = {}
        for i in range(n):
            d = np.linalg.norm(X - X[i], axis=1); d[i] = 9.0
            pred = self._labels[int(d.argmin())]
            if pred == self._labels[i]:
                correct += 1
            else:
                key = f"{self._labels[i]}→{pred}"; conf[key] = conf.get(key, 0) + 1
        return {"n": n, "accuracy": correct / n, "confusions": conf}


class StateStream:
    """即時用：每次給一張盒內裁圖（或 None＝這次不判），用最近 K 次投票決定目前狀態。"""

    def __init__(self, lib: StateLibrary, votes: int = 5, min_votes: int = 3, min_margin: float | None = None):
        self.lib = lib
        self.buf: deque[str] = deque(maxlen=votes)
        self.min_votes = min_votes
        # 與第二名的距離差至少要多少才投票：CLIP 向量用 0.03，灰階縮圖的距離尺度小得多用 0.005
        self.min_margin = min_margin
        self.current: str | None = None
        self.last: StateResult = StateResult(None)

    def push(self, crop_bgr, hands_inside: bool = False) -> str | None:
        if hands_inside or crop_bgr is None or crop_bgr.size == 0:
            self.last = StateResult(self.current, reason="手在盒內，暫不判定" if hands_inside else "沒有盒子")
            return self.current
        r = self.lib.recognize(crop_bgr)
        self.last = r
        margin_min = self.min_margin
        if margin_min is None:
            margin_min = 0.03 if (self.lib.embedder and self.lib.embedder.kind == "clip") else 0.005
        if r.state is not None and r.margin >= margin_min:
            self.buf.append(r.state)
        if self.buf:
            top = max(set(self.buf), key=self.buf.count)
            if self.buf.count(top) >= self.min_votes:
                self.current = top
        return self.current

    def reset(self) -> None:
        self.buf.clear(); self.current = None
