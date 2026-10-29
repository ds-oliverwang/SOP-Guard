"""手勢（動作）分類：用 MediaPipe 21 點手部關鍵點在 1 秒視窗內的軌跡，判斷操作者正在做哪個動作。

特徵版本 2（每格、每隻手 HAND_DIM 維，兩隻手 + 雙手距離 = FRAME_DIM）：
  - 21 點相對手腕、以手掌長度正規化的座標（42）         → 手的形狀
  - 手腕相對 N_ANCHORS 個錨點框的位置（每個 2，沒框填 0）  → 在哪個物件附近
  - 手腕在畫面中的絕對位置（x/W, y/H）（2）               → 移到畫面哪裡（版本 2 新增）
  - 手是否存在（1）
  - 兩手手腕距離 / 畫面寬（1，只有一隻手時為 0）           → 雙手動作（版本 2 新增）
視窗：WINDOW 格，等距取 STEPS 個時間點串起來，加上時序池化（平均／標準差／最小／最大），再加版本 2 的
  移動統計：每隻手的平均速度、最大速度、淨位移向量、路徑長、有手比例。
分類器：RandomForest（sklearn）。模型用 joblib 存成 gesture_model.pkl，內含 feature_version；版本不符要重訓。
"""
from __future__ import annotations

from collections import deque

import numpy as np

FEATURE_VERSION = 2
MIN_HAND_RATIO = 0.3  # 視窗裡「有手」的格數比例至少要這麼多才算（拿東西、開盒時手常被遮住，太嚴會整段沒機率）
WINDOW = 30          # 視窗長度（格）；30 fps 下 = 1 秒
STEPS = 10           # 視窗內等距取幾個時間點
STRIDE = 5           # 每幾格預測一次
N_ANCHORS = 2        # 參考錨點數（例如鞋子、鞋盒）
SHAPE_DIM = 42
ANCHOR_OFF = SHAPE_DIM                       # 42..42+2N：相對錨點
ABS_OFF = SHAPE_DIM + 2 * N_ANCHORS          # 絕對位置 x/W, y/H
PRES_OFF = ABS_OFF + 2                       # 存在旗標
HAND_DIM = PRES_OFF + 1                      # = 49
FRAME_DIM = 2 * HAND_DIM + 1                 # + 雙手距離 = 99
WRIST, MIDDLE_MCP = 0, 9
PRESENCE_IDX = [PRES_OFF, HAND_DIM + PRES_OFF]
WRIST_REL_IDX = [k * HAND_DIM + ANCHOR_OFF + j for k in range(2) for j in range(2 * N_ANCHORS)]
ABS_IDX = [[k * HAND_DIM + ABS_OFF, k * HAND_DIM + ABS_OFF + 1] for k in range(2)]


def hand_frame_features(hands_xy: np.ndarray, anchor_rects, frame_size=None) -> np.ndarray:
    """hands_xy: (2, 21, 2) 像素座標，沒有的手為 NaN。anchor_rects: 長度 N_ANCHORS 的 list（[x,y,w,h] 或 None）。
    frame_size: (W, H)，絕對位置正規化用；None 時以 1920 當寬高（只在舊呼叫端未傳時發生，結果仍可用但不建議）。"""
    out = np.zeros(FRAME_DIM, np.float32)
    rects = list(anchor_rects or [])[:N_ANCHORS]
    rects += [None] * (N_ANCHORS - len(rects))
    W, H = (frame_size if frame_size else (1920.0, 1920.0))
    W = float(W) or 1.0; H = float(H) or 1.0
    wrists = []
    for k in range(2):
        pts = hands_xy[k]
        if np.isnan(pts[0, 0]):
            continue
        base = k * HAND_DIM
        wrist = pts[WRIST]
        size = float(np.linalg.norm(pts[MIDDLE_MCP] - wrist)) or 1.0
        out[base:base + SHAPE_DIM] = ((pts - wrist) / size).ravel()
        for a, r in enumerate(rects):
            if r is not None and r[2] > 0 and r[3] > 0:
                ax, ay, aw, ah = r
                out[base + ANCHOR_OFF + 2 * a] = (wrist[0] - ax) / aw - 0.5
                out[base + ANCHOR_OFF + 2 * a + 1] = (wrist[1] - ay) / ah - 0.5
        out[base + ABS_OFF] = wrist[0] / W
        out[base + ABS_OFF + 1] = wrist[1] / H
        out[base + PRES_OFF] = 1.0
        wrists.append(wrist)
    if len(wrists) == 2:
        out[FRAME_DIM - 1] = float(np.linalg.norm(wrists[0] - wrists[1])) / W
    return out


def _motion_stats(seq: np.ndarray) -> np.ndarray:
    """每隻手：平均速度、最大速度、淨位移 dx、dy、路徑長、有手比例（位置單位 = 畫面寬高比例，速度 = 每格）。"""
    out = []
    for k in range(2):
        xy = seq[:, ABS_IDX[k]]
        present = seq[:, PRESENCE_IDX[k]] > 0.5
        if present.sum() >= 2:
            p = xy[present]
            v = np.diff(p, axis=0)
            speed = np.linalg.norm(v, axis=1)
            net = p[-1] - p[0]
            out += [float(speed.mean()), float(speed.max()), float(net[0]), float(net[1]), float(speed.sum()), float(present.mean())]
        else:
            out += [0.0, 0.0, 0.0, 0.0, 0.0, float(present.mean())]
    return np.array(out, np.float32)


def window_features(seq: np.ndarray) -> np.ndarray:
    """seq: (WINDOW, FRAME_DIM) → 1 維特徵向量。"""
    idx = np.linspace(0, len(seq) - 1, STEPS).astype(int)
    sampled = seq[idx].ravel()
    pooled = np.concatenate([seq.mean(0), seq.std(0), seq.min(0), seq.max(0)])
    disp = np.abs(np.diff(seq[:, WRIST_REL_IDX], axis=0)).sum(0)
    return np.concatenate([sampled, pooled, disp, _motion_stats(seq)]).astype(np.float32)


def window_has_enough_hand(seq: np.ndarray, min_ratio: float | None = None) -> bool:
    r = MIN_HAND_RATIO if min_ratio is None else min_ratio
    # 任一格只要有一隻手就算「有手」（PRESENCE_IDX 兩欄相加可能是 2，用 max 夾成 1）
    present = np.minimum(seq[:, PRESENCE_IDX].sum(axis=1), 1.0)
    return float(present.sum()) >= len(seq) * r


class GestureClassifier:
    def __init__(self, model=None, classes: list[str] | None = None):
        self.model = model
        self.classes = classes or []

    def fit(self, X: np.ndarray, y: list[str], n_estimators: int = 300, seed: int = 0) -> None:
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced",
                                            random_state=seed, n_jobs=-1, min_samples_leaf=2)
        self.model.fit(X, y)
        self.classes = list(self.model.classes_)

    def predict_proba(self, x: np.ndarray) -> dict[str, float]:
        p = self.model.predict_proba(x.reshape(1, -1))[0]
        return {c: float(v) for c, v in zip(self.classes, p)}

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({"model": self.model, "classes": self.classes, "window": WINDOW, "steps": STEPS,
                     "n_anchors": N_ANCHORS, "feature_version": FEATURE_VERSION, "frame_dim": FRAME_DIM}, path)

    @classmethod
    def load(cls, path: str) -> "GestureClassifier":
        import joblib
        d = joblib.load(path)
        ver = d.get("feature_version", 1)
        if ver != FEATURE_VERSION or d.get("frame_dim", FRAME_DIM) != FRAME_DIM:
            raise ValueError(f"手勢模型「{path}」是特徵版本 {ver}，目前程式是版本 {FEATURE_VERSION}，請到「② 分段說明動作」重新按「訓練並產生步驟」")
        return cls(d["model"], d["classes"])


class GestureStream:
    """即時用：每格 push 一次特徵，每 STRIDE 格算一次機率，並做指數平滑。視窗裡有手的格數不夠時機率歸零。"""

    def __init__(self, clf: GestureClassifier, smooth: float = 0.5):
        self.clf = clf
        self.buf: deque[np.ndarray] = deque(maxlen=WINDOW)
        self.smooth = smooth
        self.probs: dict[str, float] = {c: 0.0 for c in clf.classes}
        self._n = 0

    def push(self, frame_feat: np.ndarray) -> dict[str, float]:
        self.buf.append(frame_feat)
        self._n += 1
        if len(self.buf) == WINDOW and self._n % STRIDE == 0:
            seq = np.stack(self.buf)
            if window_has_enough_hand(seq):
                p = self.clf.predict_proba(window_features(seq))
            else:
                p = {c: 0.0 for c in self.clf.classes}
            for c in self.probs:
                self.probs[c] = self.smooth * self.probs[c] + (1 - self.smooth) * p.get(c, 0.0)
        return self.probs

    def top(self) -> tuple[str, float]:
        if not self.probs:
            return "", 0.0
        c = max(self.probs, key=self.probs.get)
        return c, self.probs[c]
