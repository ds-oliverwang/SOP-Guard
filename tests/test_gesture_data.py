"""手勢資料建置與依片段交叉驗證（合成手部軌跡，不需要模型檔）。"""
from __future__ import annotations

import numpy as np

from gesture_classifier import FRAME_DIM, WINDOW
from gesture_data import Clip, anchor_rects_per_frame, build_dataset, clip_windows, cross_validate_by_clip


def synth_hands(n: int, motion: str, seed: int) -> np.ndarray:
    """合成一隻手：motion='lift' 手腕往上移並張開；'push' 往下移且握拳；'none' 不動。"""
    rng = np.random.RandomState(seed)
    hands = np.full((n, 2, 21, 2), np.nan, np.float32)
    base = rng.uniform(300, 700, 2)
    for i in range(n):
        t = i / max(n - 1, 1)
        wrist = base + (np.array([0, -200 * t]) if motion == "lift" else np.array([0, 200 * t]) if motion == "push" else 0)
        spread = 60 if motion == "lift" else 25 if motion == "push" else 40
        pts = wrist + rng.normal(0, 2, (21, 2)) + np.array([[np.cos(k) * spread, np.sin(k) * spread] for k in np.linspace(0, 3, 21)])
        hands[i, 0] = pts
    return hands


def make_clips(n_per_class=6, frames=75):
    clips = []
    k = 0
    for label, motion in (("拿起", "lift"), ("放下", "push"), ("其他", "none")):
        for j in range(n_per_class):
            rects = [[(200.0, 200.0, 600.0, 400.0)] * frames, [None] * frames]
            clips.append(Clip(label=label, name=f"{label}_{j}", hands=synth_hands(frames, motion, k), rects=rects)); k += 1
    return clips


def test_clip_windows_shape_and_short_clip():
    c = make_clips(1, 75)[0]
    w = clip_windows(c)
    assert w.ndim == 2 and w.shape[0] == (75 - WINDOW) // 5 + 1
    short = Clip("x", "x", synth_hands(10, "none", 1), [[None] * 10, [None] * 10])
    assert clip_windows(short).size == 0


def test_build_dataset_groups_follow_clips():
    clips = make_clips(2, 60)
    X, y, g = build_dataset(clips)
    assert len(y) == len(g) == X.shape[0] and len(set(g.tolist())) == 6
    assert set(y) == {"拿起", "放下", "其他"}


def test_cross_validate_by_clip_separates_synthetic_motions():
    X, y, g = build_dataset(make_clips(6, 75))
    r = cross_validate_by_clip(X, y, g, n_folds=3)
    assert r["folds"] == 3 and r["n"] == len(y)
    assert r["accuracy"] >= 0.9, r
    assert all(r["recall"][c] >= 0.8 for c in r["classes"]), r


def test_anchor_rects_per_frame_union_and_hold():
    det = np.array([[0, 0, 0.9, 10, 10, 50, 50], [0, 0, 0.9, 40, 40, 100, 100], [30, 1, 0.9, 0, 0, 5, 5]], np.float32)
    rects = anchor_rects_per_frame(det, ["sneaker", "box"], "sneaker", 120, max_age=60)
    assert rects[0] == (10.0, 10.0, 90.0, 90.0)     # 兩框聯集
    assert rects[60] == rects[0] and rects[61] is None   # 沿用到 max_age 格後消失
