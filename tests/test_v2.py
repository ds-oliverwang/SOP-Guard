"""v2：場景錨點／區域、object_in_zone 與 gesture 步驟條件、schema 2 驗證。"""
from __future__ import annotations

import pytest

import core_detector
from core_detector import FrameContext, SOPStateMachine
from project_store import ProjectConfigError, project_from_dict
from scene import AnchorSpec, Detection, SceneTracker, ZoneSpec, rect_overlap_ratio, resolve_zone

FPS = 30.0


def v2_config(**over):
    cfg = {
        "schema_version": 2, "project_name": "v2", "video_file": "demo.mp4", "resolution": [1080, 1920], "fps": FPS,
        "skip_alert_sec": 1.0, "cycle_arm_clear_sec": 0.5, "reset_zone": None, "reset_dwell_sec": 1.0,
        "work_area": None, "detect_every_n_frames": 15, "gesture_model": "gesture_model.pkl", "gesture_anchors": ["shoes", "box"],
        "anchors": [
            {"name": "box", "prompt": "cardboard box", "min_conf": 0.25, "pick": "largest", "max_count": 1},
            {"name": "shoes", "prompt": "sneaker", "min_conf": 0.2, "pick": "all", "max_count": 2},
        ],
        "zones": [{"name": "shoes_zone", "anchor": "shoes"}, {"name": "box_zone", "anchor": "box"},
                  {"name": "label_zone", "anchor": "box", "rel": [0.5, 0.0, 0.5, 0.5]}],
        "steps": [
            {"step_index": 0, "step_name": "拿鞋", "type": "hand_dwell", "zone": "shoes_zone", "dwell_time_sec": 0.4, "timeout_sec": 30.0},
            {"step_index": 1, "step_name": "放鞋入盒", "type": "object_in_zone", "object": "shoes", "zone": "box_zone",
             "min_overlap": 0.6, "count": "any", "dwell_time_sec": 0.5, "timeout_sec": 30.0},
            {"step_index": 2, "step_name": "確認尺碼", "type": "gesture", "label": "尺码确认", "min_prob": 0.6,
             "dwell_time_sec": 0.5, "timeout_sec": 30.0},
        ],
    }
    cfg.update(over)
    return cfg


class Clock:
    t = 0.0


def feed(fsm, ctx: FrameContext, seconds: float, clock: Clock):
    for _ in range(int(round(seconds * FPS))):
        fsm.update(None, ctx, clock.t)
        clock.t += 1.0 / FPS


BOX = (100.0, 1000.0, 600.0, 500.0)
SHOE_A = (150.0, 300.0, 300.0, 150.0)
SHOE_B = (150.0, 480.0, 300.0, 150.0)
SHOE_IN_BOX = (200.0, 1100.0, 300.0, 150.0)


def anchors(box=True, shoes=(SHOE_A, SHOE_B)):
    return {"box": [BOX] if box else [], "shoes": list(shoes)}


# ---------------------------------------------------------------- scene

def test_scene_tracker_pick_and_age():
    tr = SceneTracker([AnchorSpec("box", "cardboard box", min_conf=0.3, pick="largest"),
                       AnchorSpec("shoes", "sneaker", pick="all", max_count=2, max_age_sec=1.0)], work_area=(0, 0, 800, 1920))
    dets = [Detection("cardboard box", 0.9, (0, 0, 100, 100)), Detection("cardboard box", 0.5, (0, 0, 400, 400)),
            Detection("cardboard box", 0.2, (0, 0, 900, 900)),      # 低於 min_conf
            Detection("sneaker", 0.6, (10, 500, 110, 600)), Detection("sneaker", 0.6, (10, 300, 110, 400)),
            Detection("sneaker", 0.6, (900, 300, 1000, 400))]        # 在 work_area 外
    tr.update_detections(dets, 0.0)
    a = tr.anchors_at(0.5)
    assert a["box"] == [(0, 0, 400, 400)]
    assert a["shoes"] == [(10, 300, 100, 100), (10, 500, 100, 100)]   # 依 y 排序
    assert tr.anchors_at(1.5)["shoes"] == []                          # 過期
    assert tr.anchors_at(1.5)["box"] == [(0, 0, 400, 400)]            # box 的 max_age 是 2.0


def test_anchor_exclude_containing():
    tr = SceneTracker([AnchorSpec("box", "cardboard box", pick="largest", exclude_containing=["shoes"]),
                       AnchorSpec("shoes", "sneaker", pick="all", max_count=2)])
    dets = [Detection("cardboard box", 0.6, (0, 0, 500, 500)),        # 包住兩隻鞋 → 是鞋子底下的紙板，不算
            Detection("cardboard box", 0.5, (600, 600, 900, 900)),    # 真正的鞋盒
            Detection("sneaker", 0.6, (50, 50, 250, 150)), Detection("sneaker", 0.6, (50, 200, 250, 300))]
    tr.update_detections(dets, 0.0)
    assert tr.anchors_at(0.0)["box"] == [(600, 600, 300, 300)]


def test_resolve_zone_rel_and_index():
    anchors_ = {"box": [(100, 1000, 600, 500)], "shoes": [SHOE_A, SHOE_B]}
    assert resolve_zone(ZoneSpec("z", anchor="box", rel=[0.5, 0.0, 0.5, 0.5]), anchors_) == [(400, 1000, 300, 250)]
    assert resolve_zone(ZoneSpec("z", anchor="shoes", index=1), anchors_) == [SHOE_B]
    assert resolve_zone(ZoneSpec("z", anchor="shoes"), {"shoes": []}) == []
    assert resolve_zone(ZoneSpec("z", fixed=[1, 2, 3, 4]), {}) == [(1, 2, 3, 4)]


def test_rect_overlap_ratio():
    assert rect_overlap_ratio((200, 1100, 300, 150), BOX) == 1.0
    assert rect_overlap_ratio((0, 0, 100, 100), BOX) == 0.0
    assert abs(rect_overlap_ratio((0, 1000, 200, 100), BOX) - 0.5) < 1e-9


# ---------------------------------------------------------------- 狀態機 v2

def test_hand_dwell_follows_moving_anchor():
    p = project_from_dict(v2_config())
    fsm = SOPStateMachine(p, sound_enabled=False); clk = Clock()
    hand_on_shoe = [(300.0, 375.0)]
    feed(fsm, FrameContext(hand_points=hand_on_shoe, anchors=anchors()), 0.5, clk)
    assert fsm.current_step_idx == 1
    # 錨點不在畫面 → 條件不成立，不推進也不誤判
    fsm2 = SOPStateMachine(p, sound_enabled=False); clk2 = Clock()
    feed(fsm2, FrameContext(hand_points=hand_on_shoe, anchors=anchors(shoes=())), 1.0, clk2)
    assert fsm2.current_step_idx == 0


def test_object_in_zone_step():
    p = project_from_dict(v2_config())
    fsm = SOPStateMachine(p, sound_enabled=False); clk = Clock()
    feed(fsm, FrameContext(hand_points=[(300.0, 375.0)], anchors=anchors()), 0.5, clk)
    assert fsm.current_step_idx == 1
    feed(fsm, FrameContext(hand_points=[], anchors=anchors(shoes=(SHOE_A, SHOE_IN_BOX))), 0.6, clk)
    assert fsm.current_step_idx == 2
    # count=all 時一隻在外面就不算
    p_all = project_from_dict(v2_config())
    p_all.steps[1].count = "all"
    fsm3 = SOPStateMachine(p_all, sound_enabled=False); clk3 = Clock()
    feed(fsm3, FrameContext(hand_points=[(300.0, 375.0)], anchors=anchors()), 0.5, clk3)
    feed(fsm3, FrameContext(hand_points=[], anchors=anchors(shoes=(SHOE_A, SHOE_IN_BOX))), 1.0, clk3)
    assert fsm3.current_step_idx == 1


def test_gesture_step_and_skip():
    p = project_from_dict(v2_config())
    fsm = SOPStateMachine(p, sound_enabled=False); clk = Clock()
    feed(fsm, FrameContext(hand_points=[(300.0, 375.0)], anchors=anchors()), 0.5, clk)
    feed(fsm, FrameContext(anchors=anchors(shoes=(SHOE_A, SHOE_IN_BOX))), 0.6, clk)
    assert fsm.current_step_idx == 2
    feed(fsm, FrameContext(anchors=anchors(), gesture_probs={"尺码确认": 0.8, "鞋垫": 0.1}), 0.6, clk)
    assert fsm.cycle_count == 1 and fsm.current_step_idx == 0
    # 從頭就做出「確認尺碼」手勢 ≥ skip_alert_sec → 跳步警告
    fsm2 = SOPStateMachine(p, sound_enabled=False); clk2 = Clock()
    feed(fsm2, FrameContext(anchors=anchors(), gesture_probs={"尺码确认": 0.9}), 1.1, clk2)
    assert fsm2.alert_active and fsm2.alert_message == "遺漏步驟：拿鞋、放鞋入盒"


def test_snapshot_reports_resolved_zones():
    p = project_from_dict(v2_config())
    fsm = SOPStateMachine(p, sound_enabled=False)
    fsm.update(None, FrameContext(anchors=anchors()), 0.0)
    z = fsm.snapshot()["zones"]
    assert z[0] == [SHOE_A, SHOE_B] and z[1] == [BOX] and z[2] == []


# ---------------------------------------------------------------- 清單模式

def checklist_config():
    return v2_config(cycle_end={"type": "object_absent", "object": "shoes", "hold_sec": 1.0}, cycle_timeout_sec=60.0)


def test_checklist_unordered_steps_and_cycle_end():
    p = project_from_dict(checklist_config())
    assert p.mode == "checklist"
    fsm = SOPStateMachine(p, sound_enabled=False); clk = Clock()
    # 順序故意顛倒：先手勢、再放鞋、再碰鞋子
    feed(fsm, FrameContext(anchors=anchors(), gesture_probs={"尺码确认": 0.8}), 0.6, clk)
    feed(fsm, FrameContext(anchors=anchors(shoes=(SHOE_A, SHOE_IN_BOX))), 0.6, clk)
    feed(fsm, FrameContext(hand_points=[(300.0, 375.0)], anchors=anchors()), 0.5, clk)
    assert fsm.observed == {0, 1, 2} and fsm.alert_active is False and fsm.cycle_count == 0
    feed(fsm, FrameContext(anchors=anchors(shoes=())), 1.1, clk)      # 鞋子離開 ≥ hold_sec → 一輪結束
    assert fsm.cycle_count == 1 and fsm.bad_cycle_count == 0 and fsm.observed == set() and fsm.armed is False
    feed(fsm, FrameContext(anchors=anchors()), 0.6, clk)              # 鞋子回來 → 重新武裝
    assert fsm.armed is True


def test_checklist_reports_missing_at_cycle_end():
    p = project_from_dict(checklist_config())
    fsm = SOPStateMachine(p, sound_enabled=False); clk = Clock()
    feed(fsm, FrameContext(hand_points=[(300.0, 375.0)], anchors=anchors()), 0.5, clk)
    feed(fsm, FrameContext(anchors=anchors(shoes=())), 1.1, clk)
    assert fsm.alert_active and fsm.alert_type == "alert_missing"
    assert fsm.alert_message == "漏做：放鞋入盒、確認尺碼"
    assert fsm.bad_cycle_count == 1 and fsm.cycle_count == 0
    fsm.reset()
    assert fsm.alert_active is False and fsm.observed == set()


def test_checklist_accumulates_across_interruptions():
    p = project_from_dict(checklist_config())
    fsm = SOPStateMachine(p, sound_enabled=False); clk = Clock()
    for _ in range(3):   # 每次只碰 0.2 秒、中間離開 0.5 秒，累計 0.6 ≥ 0.4
        feed(fsm, FrameContext(hand_points=[(300.0, 375.0)], anchors=anchors()), 0.2, clk)
        feed(fsm, FrameContext(anchors=anchors()), 0.5, clk)
    assert 0 in fsm.observed


def test_checklist_short_cycle_is_discarded():
    cfg = checklist_config(); cfg["cycle_min_sec"] = 10.0
    p = project_from_dict(cfg)
    fsm = SOPStateMachine(p, sound_enabled=False); clk = Clock()
    feed(fsm, FrameContext(hand_points=[(300.0, 375.0)], anchors=anchors()), 0.5, clk)   # 只做 0.5 秒
    feed(fsm, FrameContext(anchors=anchors(shoes=())), 1.1, clk)                           # 就結束
    assert fsm.alert_active is False and fsm.cycle_count == 0 and fsm.bad_cycle_count == 0
    assert [e["event_type"] for e in fsm.events][-1] == "cycle_discarded"
    assert fsm.armed is False and fsm.observed == set()


def test_checklist_not_required_step_does_not_alert():
    cfg = checklist_config(); cfg["steps"][2]["required"] = False
    p = project_from_dict(cfg)
    fsm = SOPStateMachine(p, sound_enabled=False); clk = Clock()
    feed(fsm, FrameContext(hand_points=[(300.0, 375.0)], anchors=anchors()), 0.5, clk)
    feed(fsm, FrameContext(anchors=anchors(shoes=(SHOE_A, SHOE_IN_BOX))), 0.6, clk)
    feed(fsm, FrameContext(anchors=anchors(shoes=())), 1.1, clk)
    assert fsm.cycle_count == 1 and fsm.alert_active is False


# ---------------------------------------------------------------- 驗證

def test_v2_validation_errors():
    with pytest.raises(ProjectConfigError):
        project_from_dict(v2_config(zones=[{"name": "shoes_zone", "anchor": "nope"}]))
    cfg = v2_config(); cfg["steps"][0]["zone"] = "missing"
    with pytest.raises(ProjectConfigError):
        project_from_dict(cfg)
    cfg = v2_config(gesture_model=None)
    with pytest.raises(ProjectConfigError) as exc:
        project_from_dict(cfg)
    assert "gesture_model" in str(exc.value)
    cfg = v2_config(); cfg["steps"][1]["object"] = "ghost"
    with pytest.raises(ProjectConfigError):
        project_from_dict(cfg)
