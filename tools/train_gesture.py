"""從快取（npz）＋逐格標籤（csv: frame,label）訓練手勢分類器，並用「留一輪」交叉驗證回報準確率。

usage: python tools/train_gesture.py <features.npz> <labels.csv> <out_model.pkl>
       [--folds "0-56,56-88,88-120,120-152"] [--anchor sneaker] [--min-purity 0.7] [--drop 等待]
labels.csv 的 label 是每格一個字串；視窗的標籤取視窗內最多的那個，純度 < min-purity 的視窗丟掉。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gesture_classifier import (GestureClassifier, WINDOW, STRIDE, hand_frame_features, window_features,  # noqa: E402
                                window_has_enough_hand)


def anchor_rect_per_frame(det: np.ndarray, labels: list[str], prompt: str, n_frames: int, every: int,
                          min_conf: float = 0.2, max_age: int = 60) -> list:
    """每格的錨點框（多個框取聯集），沒偵測到時沿用上一次（最多 max_age 格）。"""
    li = labels.index(prompt) if prompt in labels else -1
    rects = [None] * n_frames
    last = None; last_at = -10 ** 9
    by_frame: dict[int, list] = {}
    for row in det:
        f, l, c, x1, y1, x2, y2 = row
        if int(l) == li and c >= min_conf:
            by_frame.setdefault(int(f), []).append((x1, y1, x2, y2))
    for f in range(n_frames):
        if f in by_frame:
            b = np.array(by_frame[f])
            x1, y1 = b[:, 0].min(), b[:, 1].min(); x2, y2 = b[:, 2].max(), b[:, 3].max()
            last = (float(x1), float(y1), float(x2 - x1), float(y2 - y1)); last_at = f
        rects[f] = last if (f - last_at) <= max_age else None
    return rects


def build_windows(hands: np.ndarray, rect_lists: list, frame_labels: list[str], min_purity: float, drop: set[str], frame_size=None):
    """rect_lists: 每個錨點一個「每格矩形」list；每格組成 [rect_anchor0, rect_anchor1, ...]。"""
    n = len(hands)
    feats = np.stack([hand_frame_features(hands[i], [rl[i] for rl in rect_lists], frame_size) for i in range(n)])
    X, y, centers = [], [], []
    for end in range(WINDOW, n + 1, STRIDE):
        seg = feats[end - WINDOW:end]
        if not window_has_enough_hand(seg):   # 視窗裡有手的格數不到一半 → 跳過
            continue
        labs = frame_labels[end - WINDOW:end]
        lab, cnt = Counter(labs).most_common(1)[0]
        if cnt / WINDOW < min_purity or lab in drop:
            continue
        X.append(window_features(seg)); y.append(lab); centers.append(end - WINDOW // 2)
    return np.array(X), y, np.array(centers)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz"); ap.add_argument("labels"); ap.add_argument("out")
    ap.add_argument("--folds", default="0-56,56-88,88-120,120-152")
    ap.add_argument("--anchors", default="sneaker,cardboard box", help="手勢特徵的參考錨點 prompt，逗號分隔，順序要與專案 gesture_anchors 一致")
    ap.add_argument("--min-purity", type=float, default=0.7)
    ap.add_argument("--drop", default="")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=False)
    hands, det, labels = d["hands"], d["det"], [str(s) for s in d["labels"]]
    fps, W, H, every = [float(v) for v in d["meta"]]
    n = len(hands)
    frame_labels = ["等待"] * n
    with open(args.labels, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            fi = int(row["frame"])
            if 0 <= fi < n:
                frame_labels[fi] = row["label"]
    rect_lists = [anchor_rect_per_frame(det, labels, p.strip(), n, int(every)) for p in args.anchors.split(",") if p.strip()]
    drop = set(s for s in args.drop.split(",") if s)
    X, y, centers = build_windows(hands, rect_lists, frame_labels, args.min_purity, drop, frame_size=(W, H))
    print(f"windows={len(y)} dims={X.shape[1] if len(X) else 0} classes={dict(Counter(y))}")

    folds = []
    for part in args.folds.split(","):
        a, b = part.split("-"); folds.append((float(a) * fps, float(b) * fps))
    fold_id = np.array([next((k for k, (a, b) in enumerate(folds) if a <= c < b), -1) for c in centers])
    y = np.array(y)
    classes = sorted(set(y))
    conf = {c: Counter() for c in classes}
    correct = 0; total = 0
    for k in range(len(folds)):
        tr, te = fold_id != k, fold_id == k
        if te.sum() == 0 or tr.sum() == 0:
            continue
        clf = GestureClassifier(); clf.fit(X[tr], list(y[tr]))
        pred = clf.model.predict(X[te])
        for t, p in zip(y[te], pred):
            conf[t][p] += 1; correct += int(t == p); total += 1
        acc_k = (pred == y[te]).mean()
        print(f"fold {k} ({folds[k][0] / fps:.0f}-{folds[k][1] / fps:.0f}s): test={te.sum()} acc={acc_k:.2f}")
    print(f"留一輪交叉驗證 整體正確率 = {correct}/{total} = {correct / max(total, 1):.2f}")
    print("混淆矩陣（列=真實, 欄=預測）")
    print("            " + " ".join(f"{c:>8s}" for c in classes))
    for c in classes:
        row = conf[c]; tot = sum(row.values())
        rec = row[c] / tot if tot else 0.0
        print(f"{c:>10s}  " + " ".join(f"{row[p]:8d}" for p in classes) + f"   召回率={rec:.2f} (n={tot})")
    clf = GestureClassifier(); clf.fit(X, list(y))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    clf.save(args.out)
    print(f"已用全部資料訓練並存檔：{args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
