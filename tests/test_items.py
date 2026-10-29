"""物件版：種子檔讀寫、item 步驟在清單模式的判定、BoxStateRunner 掛物品偵測器、apply_item_steps 寫設定。"""
from __future__ import annotations

import json
import os

import numpy as np

import training
from core_detector import BoxStateRunner, FrameContext, SOPStateMachine
from item_recognizer import ItemHit, ItemSeeds
from project_store import load_project, project_from_dict


def _project(tmp_path, steps):
    cfg = {"schema_version": 2, "project_name": "i", "video_file": "demo.avi", "resolution": [400, 240], "fps": 30.0,
           "anchors": [{"name": "box", "prompt": "cardboard box", "max_age_sec": 1.0}], "zones": [], "state_anchor": "box", "state_every_n_frames": 1,
           "cycle_end": {"type": "object_absent", "object": "box", "hold_sec": 0.5}, "cycle_min_sec": 0.0, "steps": steps}
    return project_from_dict(cfg, str(tmp_path))


def test_item_seeds_persist(tmp_path):
    s = ItemSeeds(str(tmp_path))
    s.add_ref(120); s.add_ref(30); s.add_seed("標籤", 150, (10, 20, 100, 60)); s.add_seed("鞋子", 200, (0, 0, 300, 200)); s.add_seed("標籤", 900, (12, 22, 98, 58))
    s2 = ItemSeeds(str(tmp_path))
    assert s2.refs == [30, 120] and s2.order == ["標籤", "鞋子"] and s2.counts() == {"標籤": 2, "鞋子": 1}
    s2.remove_item("鞋子")
    assert ItemSeeds(str(tmp_path)).order == ["標籤"]


def test_item_steps_checklist_seen_once_then_box_gone(tmp_path):
    p = _project(tmp_path, [{"step_index": 0, "step_name": "標籤", "type": "item", "label": "標籤", "dwell_time_sec": 0.2, "timeout_sec": 60},
                           {"step_index": 1, "step_name": "鞋子", "type": "item", "label": "鞋子", "dwell_time_sec": 0.2, "timeout_sec": 60}])
    fsm = SOPStateMachine(p, sound_enabled=False); t = 0.0
    box = {"box": [(50, 40, 200, 150)]}

    def feed(items, sec, present=True):
        nonlocal t
        for _ in range(int(sec * 30)):
            fsm.update(None, FrameContext(anchors=box if present else {}, items_present=set(items)), t); t += 1 / 30
    feed([], 1.0)                      # 空盒
    feed(["標籤"], 0.5)                # 標籤放進去、看到了
    feed(["鞋子"], 0.5)                # 鞋子蓋住標籤：只看到鞋子也沒關係（一輪內看過就算）
    assert fsm.observed == {0, 1}
    feed([], 1.0, present=False)       # 盒子拿走 → 一輪結束
    assert fsm.cycle_count == 1 and fsm.bad_cycle_count == 0
    feed([], 1.0); feed(["鞋子"], 0.5); feed([], 1.0, present=False)   # 第二輪漏放標籤
    assert fsm.bad_cycle_count == 1 and fsm.alert_active and "標籤" in fsm.alert_message


def test_closed_box_counts_as_absent(tmp_path):
    """蓋著的盒子 YOLO 仍認成盒子；box_open=False 時結束條件（object_absent）要成立。"""
    p = _project(tmp_path, [{"step_index": 0, "step_name": "標籤", "type": "item", "label": "標籤", "dwell_time_sec": 0.2, "timeout_sec": 60}])
    fsm = SOPStateMachine(p, sound_enabled=False); t = 0.0
    box = {"box": [(50, 40, 200, 150)]}

    def feed(items, sec, box_open=True):
        nonlocal t
        for _ in range(int(sec * 30)):
            fsm.update(None, FrameContext(anchors=box, items_present=set(items), box_open=box_open), t); t += 1 / 30
    feed([], 1.0); feed(["標籤"], 0.5)
    assert 0 in fsm.observed
    feed([], 1.0, box_open=False)          # 蓋上了：盒子還在畫面上，但視同拿走 → 一輪結束
    assert fsm.cycle_count == 1
    feed([], 1.0, box_open=False)          # 蓋著放著不動：不會開始新的一輪
    assert not fsm.armed


class _FakeDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, crop):
        self.calls += 1
        return [ItemHit("標籤", 0.9, (5, 5, 20, 10))] if crop.mean() > 100 else []


def test_runner_runs_item_detector_on_judged_frames(tmp_path):
    p = _project(tmp_path, [])
    runner = BoxStateRunner(p); runner.items = _FakeDetector()
    assert runner.enabled
    bright = np.full((240, 400, 3), 200, np.uint8); dark = np.zeros((240, 400, 3), np.uint8)
    anchors = {"box": [(100, 60, 200, 120)]}
    for _ in range(runner._settle + 1):
        runner.update(bright, anchors, hand_points=[])
    assert runner.items_present == {"標籤": 0.9} and runner.item_hits[0].rect == (5, 5, 20, 10)
    calls = runner.items.calls
    runner.update(dark, anchors, hand_points=[(150.0, 100.0), (160.0, 110.0), (5.0, 5.0), (5.0, 5.0)])   # 手在盒內：物品照偵測（看得到就算）
    assert runner.items.calls == calls + 1 and runner.items_present == {} and "手在盒內" in runner.stream.last.reason
    runner.update(dark, anchors, hand_points=[])
    assert runner.items_present == {}
    runner.update(dark, {}, hand_points=[])          # 盒子不見 → 清空
    assert runner.items_present == {} and runner.item_hits == []


def test_apply_item_steps_writes_config(tmp_path):
    cfg = {"schema_version": 2, "project_name": "j", "video_file": "demo.avi", "resolution": [400, 240], "fps": 30.0, "anchors": [], "zones": [], "steps": [],
           "state_every_n_frames": 10}
    d = tmp_path / "j"; d.mkdir()
    with open(d / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    p = load_project(str(d))
    names = training.apply_item_steps(p, ["標籤", "鞋子"])
    p2 = load_project(str(d))
    assert names == ["標籤", "鞋子"]
    assert [(s.type, s.label, s.required) for s in p2.steps] == [("item", "標籤", True), ("item", "鞋子", True)]
    assert p2.cycle_end.type == "object_absent" and p2.cycle_end.object == "box" and any(a.name == "box" for a in p2.anchors)
    assert p2.state_every_n_frames == 5 and p2.cycle_min_sec == 5.0


# ---------------------------------------------------------------- 看整張桌面（固定範圍）

def _fixed_project(tmp_path, names, box_labels=()):
    cfg = {"schema_version": 2, "project_name": "f", "video_file": "demo.avi", "resolution": [400, 240], "fps": 30.0,
           "anchors": [{"name": "box", "prompt": "blue box", "max_age_sec": 1.0}], "zones": [], "state_anchor": "box", "state_fixed": [0, 0, 400, 240],
           "state_every_n_frames": 1, "cycle_arm_clear_sec": 0.2,
           "cycle_end": {"type": "object_absent", "object": "box", "hold_sec": 0.5}, "cycle_min_sec": 0.0,
           "steps": [{"step_index": i, "step_name": n, "type": "item", "label": n, "dwell_time_sec": 0.2, "timeout_sec": 60} for i, n in enumerate(names)]}
    s = ItemSeeds(str(tmp_path)); s.set_region([0, 0, 400, 240])
    for n in names:
        s.add_seed(n, 0, (0, 0, 10, 10))
    for n in box_labels:
        s.set_box_label(n, True)
    return project_from_dict(cfg, str(tmp_path))


def test_item_seeds_region_and_box_labels_persist(tmp_path):
    s = ItemSeeds(str(tmp_path)); s.add_seed("藍盒", 10, (0, 0, 50, 40)); s.add_seed("鞋舌翻開", 20, (0, 0, 50, 40))
    s.set_region((5, 6, 300, 200)); s.set_box_label("藍盒", True); s.set_box_label("沒這個", True)
    s2 = ItemSeeds(str(tmp_path))
    assert s2.region == [5, 6, 300, 200] and s2.box_labels == ["藍盒"]
    s2.remove_item("藍盒"); s2.set_region(None)
    s3 = ItemSeeds(str(tmp_path))
    assert s3.box_labels == [] and s3.region is None


class _TableDetector:
    """假模型：亮圖＝桌上看到「鞋子在盒上」（盒子的樣子）、暗圖＝什麼都沒有。"""
    def detect(self, crop):
        return [ItemHit("鞋子在盒上", 0.8, (10, 10, 50, 30))] if crop.mean() > 100 else []


def test_fixed_runner_detects_without_box_and_reports_box_seen(tmp_path):
    p = _fixed_project(tmp_path, ["鞋舌翻開", "鞋子在盒上"], box_labels=["鞋子在盒上"])
    runner = BoxStateRunner(p); runner.items = _TableDetector()
    assert runner.fixed_items and runner.box_labels == {"鞋子在盒上"}
    bright = np.full((240, 400, 3), 200, np.uint8); dark = np.zeros((240, 400, 3), np.uint8)
    runner.update(bright, {}, hand_points=[(1.0, 1.0)] * 4, anchor_age=None)      # 沒有錨點、手在畫面上：照樣偵測
    assert runner.items_present == {"鞋子在盒上": 0.8} and runner.box_seen and runner.region == (0, 0, 400, 240)
    runner.update(dark, {}, hand_points=[], anchor_age=9.0)
    assert runner.items_present == {} and not runner.box_seen


def test_fixed_checklist_round_starts_before_box_and_ends_after_push(tmp_path):
    """現場流程：翻鞋舌檢查（盒子還沒放上來）→ 拿盒子 → 鞋子放到盒上（YOLO 找不到盒子，但模型看得到盒子的樣子）→ 推走。"""
    p = _fixed_project(tmp_path, ["鞋舌翻開", "藍盒", "鞋子在盒上"], box_labels=["藍盒", "鞋子在盒上"])
    fsm = SOPStateMachine(p, sound_enabled=False); t = 0.0
    box = {"box": [(50, 40, 200, 150)]}

    def feed(items, sec, present=False, seen=False):
        nonlocal t
        for _ in range(int(sec * 30)):
            fsm.update(None, FrameContext(anchors=box if present else {}, items_present=set(items), box_seen=seen), t); t += 1 / 30
    feed([], 1.0)
    feed(["鞋舌翻開"], 0.6)                             # 盒子還不在：也算這一輪開始，不會被當成「盒子被拿走」而結束
    assert fsm.cycle_active and 0 in fsm.observed and fsm.cycle_count == 0
    feed(["藍盒"], 1.0, present=True, seen=True)
    feed(["鞋子在盒上"], 1.0, present=False, seen=True)   # YOLO 丟了盒子、模型還看得到盒子的樣子 → 不算推走
    assert fsm.cycle_count == 0 and fsm.observed == {0, 1, 2}
    feed([], 1.0)                                       # 真的推走
    assert fsm.cycle_count == 1 and fsm.bad_cycle_count == 0
    feed(["鞋舌翻開"], 1.0)                             # 下一雙：推走後 0.2 秒就能開始
    feed(["鞋子在盒上"], 1.0, present=True, seen=True)
    feed([], 1.0)
    assert fsm.bad_cycle_count == 1 and "藍盒" in fsm.alert_message and "鞋舌翻開" not in fsm.alert_message
    fsm.reset()
    # 輪與輪之間空檔時盒子閃一下（YOLO 誤判、別處的藍色東西）不能擋住下一輪：現場影片第 6 輪因此整輪沒開始
    feed([], 1.0); feed([], 0.3, present=True); feed([], 30.0)
    feed(["鞋舌翻開"], 0.6)
    assert fsm.cycle_active and 0 in fsm.observed
    feed(["藍盒"], 1.0, present=True, seen=True); feed(["鞋子在盒上"], 1.0, seen=True); feed([], 1.0)
    assert fsm.cycle_count == 2 and fsm.bad_cycle_count == 1


def test_apply_item_steps_fixed_region_writes_config(tmp_path):
    cfg = {"schema_version": 2, "project_name": "k", "video_file": "demo.avi", "resolution": [400, 240], "fps": 30.0, "anchors": [], "zones": [], "steps": []}
    d = tmp_path / "k"; d.mkdir()
    with open(d / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    s = ItemSeeds(str(d)); s.set_region([10, 20, 300, 200]); s.add_seed("藍盒", 3, (0, 0, 9, 9)); s.set_box_label("藍盒", True)
    p = load_project(str(d))
    training.apply_item_steps(p, ["藍盒"])
    p2 = load_project(str(d))
    assert p2.state_fixed == [10, 20, 300, 200] and p2.state_anchor == "box" and p2.cycle_arm_clear_sec == 0.2
    assert next(a for a in p2.anchors if a.name == "box").max_age_sec == 1.0
    s.set_region(None); training.apply_item_steps(load_project(str(d)), ["藍盒"])
    assert load_project(str(d)).state_fixed is None


def test_build_dataset_fixed_propagates_near_seed_and_writes_background(tmp_path):
    import shutil
    import cv2
    import item_recognizer as ir
    d = tmp_path / "v"; d.mkdir()
    path = str(d / "demo.avi"); wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (320, 240))
    for n in range(120):
        fr = np.full((240, 320, 3), 40, np.uint8)
        if 50 <= n <= 70:                                                 # 第 50～70 格桌上有一個亮塊（物品）
            cv2.rectangle(fr, (100, 80), (180, 140), (200, 220, 230), -1); cv2.circle(fr, (140, 110), 15, (30, 30, 30), -1)
        wr.write(fr)
    wr.release()
    cfg = {"schema_version": 2, "project_name": "v", "video_file": "demo.avi", "resolution": [320, 240], "fps": 30.0,
           "anchors": [{"name": "box", "prompt": "blue box", "max_age_sec": 1.0}], "zones": [], "state_anchor": "box", "state_fixed": [0, 0, 320, 240], "steps": []}
    with open(d / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    p = load_project(str(d))
    s = ItemSeeds(str(d)); s.set_region([0, 0, 320, 240]); s.add_ref(10); s.add_seed("物品", 60, (100, 80, 80, 60))
    trace = {"fingerprint": "x", "prompts": ["blue box"], "fps": 30.0, "frames": 120, "rows": []}
    try:
        stats = ir.build_dataset(p, trace, s, val_fraction=0.0)
        ds = ir.dataset_dir(p)
        labelled = sorted(os.listdir(os.path.join(ds, "labels", "train")))
        imgs = sorted(os.listdir(os.path.join(ds, "images", "train")))
        assert "f000060.txt" in labelled and "f000055.txt" in labelled and "f000065.txt" in labelled   # 種子前後 2 秒內同樣東西都標到
        assert "f000030.txt" not in labelled and "f000090.txt" not in labelled                           # 亮塊不在時不標
        assert "f000010.jpg" in imgs and "f000010.txt" not in labelled                                  # 空桌面＝背景圖，沒有標籤
        assert stats["per_item"]["物品"] >= 4 and stats["work_area"] == [0, 0, 320, 240]
    finally:
        shutil.rmtree(ir.dataset_dir(p), ignore_errors=True)


def test_fixed_runner_cadence_follows_source_time(tmp_path):
    """every=5、fps 30 → 每 0.167 秒跑一次偵測：處理 30 fps 時每 5 格、處理慢到 5 fps（來源時間每格 0.2 秒）時每格都跑。"""
    p = _fixed_project(tmp_path, ["鞋子在盒上"]); p.state_every_n_frames = 5
    runner = BoxStateRunner(p); runner.items = _TableDetector(); runner.every = 5
    bright = np.full((240, 400, 3), 200, np.uint8)
    class Counter(_TableDetector):
        calls = 0
        def detect(self, crop):
            Counter.calls += 1; return super().detect(crop)
    runner.items = Counter()
    for i in range(30):                                   # 30 fps：每格 0.0333 秒
        runner.update(bright, {}, [], t_now=i / 30.0)
    fast = Counter.calls
    for i in range(30):                                   # 5 fps：每格 0.2 秒
        runner.update(bright, {}, [], t_now=1.0 + i * 0.2)
    slow = Counter.calls - fast
    assert 5 <= fast <= 7 and slow == 30


def _moving_video(path, n=90, start=(60, 80), vx=2):
    """亮塊（帶黑點紋理）每格往右移 vx 像素；第 70 格以後消失。"""
    import cv2
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (320, 240))
    for k in range(n):
        fr = np.full((240, 320, 3), 40, np.uint8)
        if k < 70:
            x = start[0] + vx * k; y = start[1]
            cv2.rectangle(fr, (x, y), (x + 60, y + 50), (200, 220, 230), -1); cv2.circle(fr, (x + 20, y + 25), 10, (30, 30, 30), -1)
            cv2.line(fr, (x + 35, y + 10), (x + 55, y + 40), (60, 60, 60), 3)
        wr.write(fr)
    wr.release()


def test_track_seed_follows_moving_item_and_stops_when_gone(tmp_path):
    import item_recognizer as ir
    path = str(tmp_path / "m.avi"); _moving_video(path)
    out = ir.track_seed(path, 30, (60 + 60, 80, 60, 50), window_sec=2.0, fps=30.0)
    frames = [f for f, _ in out]
    assert frames and frames[0] < 30 < frames[-1] and frames[-1] < 72          # 往前往後都追到，物品消失（第 70 格）就停
    f45 = next(r for f, r in out if f == 45)
    assert abs(f45[0] - (60 + 90)) <= 4 and abs(f45[1] - 80) <= 4                # 跟著移動


def test_seed_groups_auto_counts_and_dataset_uses_auto(tmp_path):
    import shutil
    import item_recognizer as ir
    d = tmp_path / "g"; d.mkdir(); _moving_video(str(d / "demo.avi"))
    cfg = {"schema_version": 2, "project_name": "g", "video_file": "demo.avi", "resolution": [320, 240], "fps": 30.0,
           "anchors": [{"name": "box", "prompt": "blue box", "max_age_sec": 1.0}], "zones": [], "state_anchor": "box", "state_fixed": [0, 0, 320, 240], "steps": []}
    with open(d / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    p = load_project(str(d))
    s = ItemSeeds(str(d)); s.set_region([0, 0, 320, 240]); s.add_ref(80)
    auto = ir.track_seed(p.video_path, 30, (120, 80, 60, 50), window_sec=1.0, fps=30.0)
    g = s.add_seed("物品", 30, (120, 80, 60, 50), auto)
    s2 = ItemSeeds(str(d))
    assert s2.counts() == {"物品": 1} and s2.auto_counts() == {"物品": len(auto)} and all(x.group == g for x in s2.seeds)
    direct, prop = ir.split_seeds(s2)
    assert len(direct) == len(auto) + 1 and prop == []
    trace = {"fingerprint": "x", "prompts": ["blue box"], "fps": 30.0, "frames": 90, "rows": []}
    try:
        stats = ir.build_dataset(p, trace, s2, val_fraction=0.0)
        assert stats["per_item"]["物品"] == len(auto) + 1                          # 只用手框＋自動框，不再額外延伸
    finally:
        shutil.rmtree(ir.dataset_dir(p), ignore_errors=True)
    i_auto = next(i for i, x in enumerate(s2.seeds) if x.auto)
    s2.remove_seed(i_auto); assert ItemSeeds(str(d)).auto_counts()["物品"] == len(auto) - 1
    i_man = next(i for i, x in enumerate(s2.seeds) if not x.auto)
    s2.remove_seed(i_man); assert ItemSeeds(str(d)).seeds == []                   # 刪手框的那一個 → 整組刪


# ---------------------------------------------------------------- 手停在物件上（hand_on_object，連續停留）

def test_hand_on_object_needs_continuous_dwell(tmp_path):
    """手在鞋子框內連續停 0.7 秒才算；零碎經過（每次 0.2 秒、中間斷 1 秒）整輪加起來也不算；中斷 0.2 秒以內算同一段。"""
    cfg = {"schema_version": 2, "project_name": "h", "video_file": "demo.avi", "resolution": [400, 240], "fps": 30.0,
           "anchors": [{"name": "box", "prompt": "blue box", "max_age_sec": 1.0}, {"name": "shoe", "prompt": "shoe", "max_count": 2, "max_age_sec": 2.5}],
           "zones": [], "state_anchor": "box", "state_fixed": [0, 0, 400, 240], "cycle_arm_clear_sec": 0.2,
           "cycle_end": {"type": "object_absent", "object": "box", "hold_sec": 0.5}, "cycle_min_sec": 0.0,
           "steps": [{"step_index": 0, "step_name": "檢查鞋舌", "type": "hand_on_object", "object": "shoe", "dwell_time_sec": 0.7, "timeout_sec": 60, "continuous": True},
                     {"step_index": 1, "step_name": "藍盒", "type": "item", "label": "藍盒", "dwell_time_sec": 0.2, "timeout_sec": 60}]}
    p = project_from_dict(cfg, str(tmp_path))
    assert p.steps[0].continuous and p.steps[0].object == "shoe"
    fsm = SOPStateMachine(p, sound_enabled=False); t = 0.0
    shoe = {"shoe": [(100, 50, 120, 80), (100, 140, 120, 80)]}
    on = [(110.0, 60.0), (130.0, 70.0), (150.0, 80.0), (140.0, 90.0)]          # 一隻手 4 點都在第一隻鞋框內
    off = [(300.0, 200.0), (310.0, 210.0), (320.0, 220.0), (330.0, 230.0)]

    def feed(pts, sec, shoes=True):
        nonlocal t
        for _ in range(int(sec * 30)):
            fsm.update(None, FrameContext(anchors=shoe if shoes else {}, hand_points=pts), t); t += 1 / 30
    for _ in range(5):                                   # 零碎經過 5 次 × 0.2 秒 = 1.0 秒，不算
        feed(on, 0.2); feed(off, 1.0)
    assert 0 not in fsm.observed and not fsm.cycle_active
    feed(on, 0.4); feed(off, 0.2); feed(on, 0.4)         # 0.4 + 中斷 0.2 + 0.4：同一段，連續 1.0 秒 → 算
    assert 0 in fsm.observed and fsm.cycle_active
    assert fsm._zones(p.steps[0], FrameContext(anchors=shoe)) == shoe["shoe"]    # 畫面上的區域＝鞋子的框


def test_apply_item_steps_hand_steps_replace_item_and_add_anchor(tmp_path):
    cfg = {"schema_version": 2, "project_name": "m", "video_file": "demo.avi", "resolution": [400, 240], "fps": 30.0, "anchors": [], "zones": [], "steps": []}
    d = tmp_path / "m"; d.mkdir()
    with open(d / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    s = ItemSeeds(str(d)); s.set_region([0, 0, 400, 240]); s.add_seed("檢查鞋舌", 3, (0, 0, 9, 9)); s.add_seed("藍盒", 4, (0, 0, 9, 9))
    s.add_hand_step("檢查鞋舌", "shoe", 0.7)
    assert ItemSeeds(str(d)).hand_steps == [{"name": "檢查鞋舌", "prompt": "shoe", "dwell": 0.7}]
    training.apply_item_steps(load_project(str(d)), s.order)
    p = load_project(str(d))
    assert [(st.type, st.step_name, st.continuous) for st in p.steps] == [("item", "藍盒", False), ("hand_on_object", "檢查鞋舌", True)]
    assert p.steps[1].object == "shoe" and p.steps[1].dwell_time_sec == 0.7
    shoe = next(a for a in p.anchors if a.name == "shoe")
    assert shoe.prompt == "shoe" and shoe.max_count == 2 and shoe.max_age_sec == 2.5
    s.remove_hand_step("檢查鞋舌"); training.apply_item_steps(load_project(str(d)), s.order)
    assert [st.type for st in load_project(str(d)).steps] == ["item", "item"]


def test_gpu_path_selection_and_cpu_fallback(monkeypatch, tmp_path):
    """確認有 GPU 時 ItemDetector 走 PyTorch + CUDA（不轉 OpenVINO）、ObjectDetector 指定 device='0'、limit_torch_threads 不壓成 1；
    沒有 GPU 時維持原本的 OpenVINO 與 CPU 行為。"""
    import torch
    import ultralytics
    import core_detector as cd
    import item_recognizer as ir
    import object_detector as od

    ov_calls = []
    predict_calls = []

    class DummyBox:
        def __init__(self):
            self.cls = torch.tensor([0])
            self.conf = torch.tensor([0.85])
            self.xyxy = torch.tensor([[10.0, 20.0, 50.0, 60.0]])

    class DummyResult:
        def __init__(self):
            self.boxes = [DummyBox()]

    class DummyYOLO:
        def __init__(self, path, task=None):
            self.path = path
            self.ckpt = {"train_args": {"imgsz": 960}}
            self.names = {0: "藍盒"}

        def set_classes(self, prompts):
            self.prompts = prompts

        def predict(self, img, **kw):
            predict_calls.append(kw)
            return [DummyResult()]

    monkeypatch.setattr(ultralytics, "YOLO", DummyYOLO)
    monkeypatch.setattr(ir, "openvino_model", lambda pt, sz: (ov_calls.append((pt, sz)) or str(tmp_path / "ov")))

    # 1. 有 GPU (CUDA available = True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    orig_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    cd.limit_torch_threads(1)
    assert torch.get_num_threads() == 4        # 有 GPU 時不壓成 1 執行緒

    det_gpu = ir.ItemDetector("dummy.pt")
    assert det_gpu.backend == "cuda" and det_gpu.device == "0" and det_gpu.half is True
    assert ov_calls == []                      # 有 GPU 時不轉 OpenVINO
    predict_calls.clear()
    hits = det_gpu.detect(np.zeros((100, 100, 3), np.uint8))
    assert len(hits) == 1 and hits[0].name == "藍盒"
    assert predict_calls[-1].get("device") == "0" and predict_calls[-1].get("half") is True

    w = tmp_path / "w.pt"; w.write_bytes(b"x")
    obj_gpu = od.ObjectDetector(str(w), ["blue box"])
    assert obj_gpu.device == "0"
    predict_calls.clear()
    obj_gpu.detect(np.zeros((100, 100, 3), np.uint8))
    assert predict_calls[-1].get("device") == "0"

    # 2. 無 GPU (CUDA available = False) → 維持原本 OpenVINO / CPU 行為
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cd.limit_torch_threads(1)
    assert torch.get_num_threads() == 1
    det_cpu = ir.ItemDetector("dummy.pt")
    assert det_cpu.backend == "openvino" and len(ov_calls) == 1
    obj_cpu = od.ObjectDetector(str(w), ["blue box"])
    assert obj_cpu.device == "cpu"
    torch.set_num_threads(orig_threads)
