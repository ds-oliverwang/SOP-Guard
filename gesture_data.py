"""手勢訓練資料：從「每段影片一個標籤」的片段建視窗、依片段分組做交叉驗證。純函式，不碰相機與模型檔。"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from gesture_classifier import STRIDE, WINDOW, GestureClassifier, hand_frame_features, window_features, window_has_enough_hand


@dataclass
class Clip:
    """一段影片的快取：hands (F, 2, 21, 2)；rects 每個錨點一個「每格矩形或 None」的 list；frame_size = (W, H)。"""
    label: str
    name: str
    hands: np.ndarray
    rects: list[list] = field(default_factory=list)
    frame_size: tuple[float, float] | None = None


def anchor_rects_per_frame(det: np.ndarray, det_labels: list[str], prompt: str, n_frames: int,
                           min_conf: float = 0.2, max_age: int = 60) -> list:
    """從快取的偵測陣列 (M,7) 做出每格的錨點矩形（多框取聯集，沒偵測到沿用上一次最多 max_age 格）。"""
    li = det_labels.index(prompt) if prompt in det_labels else -1
    by_frame: dict[int, list] = {}
    for row in det:
        f, l, c, x1, y1, x2, y2 = row
        if int(l) == li and c >= min_conf:
            by_frame.setdefault(int(f), []).append((x1, y1, x2, y2))
    rects = [None] * n_frames
    last = None; last_at = -10 ** 9
    for f in range(n_frames):
        if f in by_frame:
            b = np.array(by_frame[f])
            last = (float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].max() - b[:, 0].min()), float(b[:, 3].max() - b[:, 1].min()))
            last_at = f
        rects[f] = last if (f - last_at) <= max_age else None
    return rects


def clip_windows(clip: Clip, min_hand_ratio: float | None = None) -> np.ndarray:
    """一段片段 → 視窗特徵 (N, D)。整段都算同一個標籤。"""
    n = len(clip.hands)
    if n < WINDOW:
        return np.zeros((0, 0), np.float32)
    feats = np.stack([hand_frame_features(clip.hands[i], [rl[i] for rl in clip.rects], clip.frame_size) for i in range(n)])
    out = []
    for end in range(WINDOW, n + 1, STRIDE):
        seg = feats[end - WINDOW:end]
        if window_has_enough_hand(seg, min_hand_ratio):
            out.append(window_features(seg))
    return np.array(out, np.float32) if out else np.zeros((0, 0), np.float32)


def clips_from_segments(hands: np.ndarray, rects: list[list], segments: list[dict], other_label: str = "其他",
                        min_frames: int = WINDOW, fill_other: bool = True, frame_size=None) -> list[Clip]:
    """一支示範影片＋使用者標的分段 → 片段清單。
    segments: [{"label": "確認尺碼", "start": 120, "end": 210}, ...]（end 不含）。
    沒標到的空隙（長度 ≥ min_frames）自動當 other_label；分段重疊時以先出現者為準。"""
    n = len(hands)
    segs = sorted(({"label": s["label"], "start": max(0, int(s["start"])), "end": min(n, int(s["end"]))} for s in segments),
                  key=lambda s: s["start"])
    clips: list[Clip] = []
    cursor = 0
    k = 0
    for s in segs:
        if s["start"] < cursor:            # 與前一段重疊 → 砍掉重疊部分
            s["start"] = cursor
        if fill_other and s["start"] - cursor >= min_frames:
            clips.append(Clip(other_label, f"{other_label}_{k:03d}", hands[cursor:s["start"]], [rl[cursor:s["start"]] for rl in rects], frame_size)); k += 1
        if s["end"] - s["start"] >= min_frames:
            clips.append(Clip(s["label"], f"{s['label']}_{k:03d}", hands[s["start"]:s["end"]], [rl[s["start"]:s["end"]] for rl in rects], frame_size)); k += 1
        cursor = max(cursor, s["end"])
    if fill_other and n - cursor >= min_frames:
        clips.append(Clip(other_label, f"{other_label}_{k:03d}", hands[cursor:n], [rl[cursor:n] for rl in rects], frame_size))
    return clips


def build_dataset(clips: list[Clip]):
    """回傳 X, y, groups（groups = 片段序號，交叉驗證用）。"""
    X, y, g = [], [], []
    for k, c in enumerate(clips):
        w = clip_windows(c)
        if w.size == 0:
            continue
        X.append(w); y += [c.label] * len(w); g += [k] * len(w)
    if not X:
        return np.zeros((0, 0), np.float32), [], np.zeros(0, int)
    return np.vstack(X), y, np.array(g)


def cross_validate_by_clip(X: np.ndarray, y: list[str], groups: np.ndarray, n_folds: int = 5, seed: int = 0) -> dict:
    """把片段輪流分成 n_folds 組（每組的片段整段一起當測試），回報整體正確率、每類召回率與混淆矩陣。
    每個類別的片段都盡量平均分到各組；某類只有 1 段時那一段只會出現在一組的測試集裡。"""
    y = np.array(y)
    clip_ids = sorted(set(groups.tolist()))
    label_of = {k: y[groups == k][0] for k in clip_ids}
    rng = np.random.RandomState(seed)
    fold_of: dict[int, int] = {}
    for lab in sorted(set(label_of.values())):
        ids = [k for k in clip_ids if label_of[k] == lab]
        rng.shuffle(ids)
        for i, k in enumerate(ids):
            fold_of[k] = i % n_folds
    fold = np.array([fold_of[k] for k in groups])
    classes = sorted(set(y.tolist()))
    conf = {c: Counter() for c in classes}
    correct = total = 0
    folds_used = 0
    for f in range(n_folds):
        te = fold == f; tr = ~te
        if te.sum() == 0 or tr.sum() == 0 or len(set(y[tr].tolist())) < 2:
            continue
        clf = GestureClassifier(); clf.fit(X[tr], y[tr].tolist())
        pred = clf.model.predict(X[te])
        for t, p in zip(y[te], pred):
            conf[t][p] += 1; correct += int(t == p); total += 1
        folds_used += 1
    recall = {c: (conf[c][c] / sum(conf[c].values())) if sum(conf[c].values()) else 0.0 for c in classes}
    return {"accuracy": correct / total if total else 0.0, "n": total, "folds": folds_used,
            "classes": classes, "recall": recall, "confusion": conf,
            "clips_per_class": Counter(label_of.values()), "windows_per_class": Counter(y.tolist())}


def format_report(r: dict) -> str:
    lines = [f"片段數：{dict(r['clips_per_class'])}", f"視窗數：{dict(r['windows_per_class'])}",
             f"依片段分組交叉驗證（{r['folds']} 組）整體正確率 = {r['accuracy']:.2f}（{r['n']} 個視窗）",
             "混淆矩陣（列=真實, 欄=預測）", "            " + " ".join(f"{c:>8s}" for c in r["classes"])]
    for c in r["classes"]:
        row = r["confusion"][c]
        lines.append(f"{c:>10s}  " + " ".join(f"{row[p]:8d}" for p in r["classes"]) + f"   召回率={r['recall'][c]:.2f}")
    return "\n".join(lines)
