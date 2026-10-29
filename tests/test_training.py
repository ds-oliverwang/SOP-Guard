"""分段 → 片段、hands_absent 結束條件、一鍵訓練寫步驟（用合成資料，不碰相機與模型檔）。"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

import training
from core_detector import FrameContext, SOPStateMachine
from gesture_data import clips_from_segments
from project_store import load_project, project_from_dict
from test_gesture_data import synth_hands


def test_clips_from_segments_fills_gaps_and_trims_overlap():
    n = 400
    hands = synth_hands(n, "none", 0)
    rects = [[None] * n]
    segs = [{"label": "A", "start": 50, "end": 120}, {"label": "B", "start": 110, "end": 200}, {"label": "A", "start": 300, "end": 320}]
    clips = clips_from_segments(hands, rects, segs)
    names = [(c.label, len(c.hands)) for c in clips]
    # B 被砍掉與 A 重疊的 10 格；最後的 A 只有 20 格太短被丟，但那 20 格也不算「其他」
    assert names == [("其他", 50), ("A", 70), ("B", 80), ("其他", 100), ("其他", 80)]


def test_segment_counts_ignore_overlaps_and_detect_overlap():
    segs = [{"label": "A", "start": 0, "end": 150}, {"label": "A", "start": 26, "end": 158}, {"label": "A", "start": 34, "end": 143},
            {"label": "B", "start": 276, "end": 415}, {"label": "B", "start": 286, "end": 401}, {"label": "A", "start": 450, "end": 500}]
    assert training.segment_counts(segs) == {"A": 2, "B": 1}
    assert training.overlapping_segment(segs, 140, 200)["label"] == "A"
    assert training.overlapping_segment(segs, 158, 276) is None


def test_feature_cache_is_tied_to_the_video_file(desk_project, monkeypatch):
    """換了示範影片（即使新檔案的修改時間比快取舊）就要重抽特徵。"""
    import time as _time
    p = load_project(desk_project)
    video = os.path.join(desk_project, "demo.avi")
    with open(video, "wb") as f:
        f.write(b"A" * 100)
    calls = []

    def fake_extract(project, hand_model, yolo_weights, progress=None, force=False):
        # 模擬抽特徵本體：用真的 extract_demo_features 的快取邏輯，只把 MediaPipe 的部分換掉
        return training.extract_demo_features.__wrapped__(project, hand_model, yolo_weights, progress, force)

    fp1 = training.video_fingerprint(video)
    cache = os.path.join(desk_project, training.CACHE_FILE)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, hands=np.zeros((10, 2, 21, 2), np.float32), det=np.zeros((0, 7), np.float32),
                        labels=np.array([]), fps=np.float32(30), fingerprint=np.array(fp1))
    d = training.extract_demo_features(p, "hand.task", None)          # 指紋相同 → 直接用快取，不會去讀影片
    assert len(d["hands"]) == 10
    _time.sleep(0.01)
    with open(video, "wb") as f:                                       # 換成另一支（大小不同）
        f.write(b"B" * 200)
    os.utime(video, (0, 0))                                            # 修改時間故意設得比快取還舊
    assert training.video_fingerprint(video) != fp1
    with pytest.raises(Exception):                                     # 快取失效 → 真的去開影片（假檔案開不了）
        training.extract_demo_features(p, "hand.task", None)


def test_propose_segments_finds_later_rounds_from_first_round():
    """只標第一輪（舉手、壓下各一段），要能在後面兩輪提出同名的建議分段，位置大致對。"""
    n = 1200
    hands = synth_hands(n, "none", 3)
    truth = []
    for k, (lab, motion) in enumerate([("舉手", "lift"), ("壓下", "push")] * 3):
        s = 60 + k * 180
        hands[s:s + 90] = synth_hands(90, motion, 20 + k)
        truth.append({"label": lab, "start": s, "end": s + 90})
    data = {"hands": hands, "det": np.zeros((0, 7), np.float32), "labels": []}
    labeled = truth[:2]                      # 只標第一輪
    props = training.propose_segments(data, labeled, prob_min=0.4)
    assert props, "應該要有建議分段"
    hits = 0
    for t in truth[2:]:
        m = [p for p in props if p["label"] == t["label"] and p["start"] < t["end"] and t["start"] < p["end"]]
        if m:
            p = m[0]; overlap = min(p["end"], t["end"]) - max(p["start"], t["start"])
            if overlap >= 45:                # 至少重疊一半（1.5 秒）
                hits += 1
    assert hits >= 3, (hits, props)         # 後面 4 段至少對到 3 段
    assert all(training.overlapping_segment(labeled, p["start"], p["end"]) is None for p in props)


def test_hands_absent_cycle_end():
    cfg = {
        "schema_version": 2, "project_name": "p", "video_file": "demo.avi", "resolution": [1280, 720], "fps": 30.0,
        "cycle_end": {"type": "hands_absent", "hold_sec": 1.0}, "cycle_min_sec": 0.0, "gesture_model": "m.pkl",
        "anchors": [], "zones": [],
        "steps": [{"step_index": 0, "step_name": "動作A", "type": "gesture", "label": "A", "min_prob": 0.5, "dwell_time_sec": 0.3, "timeout_sec": 30}],
    }
    p = project_from_dict(cfg)
    fsm = SOPStateMachine(p, sound_enabled=False)
    t = 0.0
    for _ in range(15):
        fsm.update(None, FrameContext(hand_points=[(1.0, 1.0)], gesture_probs={"A": 0.9}), t); t += 1 / 30
    assert 0 in fsm.observed
    for _ in range(35):
        fsm.update(None, FrameContext(hand_points=[], gesture_probs={}), t); t += 1 / 30
    assert fsm.cycle_count == 1


@pytest.fixture
def desk_project(tmp_path):
    d = tmp_path / "desk"; (d / "gestures").mkdir(parents=True)
    cfg = {"schema_version": 2, "project_name": "desk", "video_file": "demo.avi", "resolution": [1280, 720], "fps": 30.0,
           "anchors": [], "zones": [], "steps": [], "gesture_model": None, "gesture_anchors": []}
    (d / "config.json").write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    return str(d)


def test_end_gesture_label_becomes_cycle_end(desk_project):
    p = load_project(desk_project)
    open(os.path.join(desk_project, "m.pkl"), "wb").close()   # load_project 會檢查模型檔存在
    training.apply_gesture_steps(p, ["舉手", "結束", "壓下"], "m.pkl")
    p2 = load_project(desk_project)
    assert [s.step_name for s in p2.steps] == ["舉手", "壓下"]
    assert p2.cycle_end.type == "gesture" and p2.cycle_end.label == "結束" and p2.cycle_end.dwell_time_sec == 0.5
    # 清單模式：兩步做完 → 比「結束」0.6 秒 → 一輪完成（這裡的假輪次只有 2 秒，先關掉 5 秒的最短輪次限制）
    assert p2.cycle_min_sec == 5.0
    p2.cycle_min_sec = 0.0
    fsm = SOPStateMachine(p2, sound_enabled=False); t = 0.0
    for lab in ("舉手", "壓下"):
        for _ in range(20):
            fsm.update(None, FrameContext(hand_points=[(1.0, 1.0)], gesture_probs={lab: 0.9, "結束": 0.05}), t); t += 1 / 30
    for _ in range(20):
        fsm.update(None, FrameContext(hand_points=[(1.0, 1.0)], gesture_probs={"結束": 0.9}), t); t += 1 / 30
    assert fsm.cycle_count == 1 and fsm.bad_cycle_count == 0


def test_train_project_writes_model_and_steps(desk_project, monkeypatch):
    # 合成一支 900 格的「示範影片」特徵：A 動作三段、B 動作三段，其餘不動
    n = 900
    hands = synth_hands(n, "none", 1)
    segs = []
    for k, (lab, motion) in enumerate([("舉手", "lift"), ("壓下", "push")] * 3):
        s = 60 + k * 140
        hands[s:s + 75] = synth_hands(75, motion, 10 + k)
        segs.append({"label": lab, "start": s, "end": s + 75})
    p = load_project(desk_project)
    training.save_segments(p, segs)
    monkeypatch.setattr(training, "extract_demo_features",
                        lambda project, hm, yw, progress=None, force=False: {"hands": hands, "det": np.zeros((0, 7), np.float32), "labels": [], "fps": 30.0})
    stages = []
    r = training.train_project(desk_project, "hand.task", None, progress=lambda st, d, t: stages.append(st))
    assert r.model_path and os.path.isfile(r.model_path)
    assert set(r.labels) == {"舉手", "壓下", "其他"}
    assert r.accuracy >= 0.8, r.report
    p2 = load_project(desk_project)
    assert [s.step_name for s in p2.steps] == ["舉手", "壓下"]
    assert all(s.type == "gesture" and s.label == s.step_name for s in p2.steps)
    assert p2.cycle_end is not None and p2.cycle_end.type == "hands_absent" and p2.gesture_model == "gesture_model.pkl"
    assert "完成" in stages
