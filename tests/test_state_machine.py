"""規格第 8 節第一階段測試表。不用相機、不用 mediapipe、不寫檔、不出聲。"""
from __future__ import annotations

import os

import pytest

import core_detector
from conftest import (IN_RESET, IN_ROI0, IN_ROI1, NO_HAND, RESET_ZONE, ROI0, ROI1, feed, sample_config,
                      write_project)
from project_store import ProjectConfigError, load_project


def _all_hand_only(project):
    for s in project.steps:
        s.trigger_mode = "hand_only"
        s.visual_match_min = None
    return project


def _events(fsm, kind):
    return [e for e in fsm.events if e["event_type"] == kind]


def test_normal_cycle(project, make_fsm, clock):
    fsm = make_fsm(_all_hand_only(project))
    feed(fsm, IN_ROI0, 0.5, clock)
    feed(fsm, NO_HAND, 0.3, clock)
    feed(fsm, IN_ROI1, 0.6, clock)
    assert fsm.cycle_count == 1
    assert fsm.current_step_idx == 0
    assert fsm.armed is False
    assert fsm.alert_active is False
    assert not _events(fsm, "alert_skip") and not _events(fsm, "alert_timeout")


def test_arming_after_cycle(project, make_fsm, clock):
    fsm = make_fsm(_all_hand_only(project))
    feed(fsm, IN_ROI0, 0.5, clock)
    feed(fsm, NO_HAND, 0.3, clock)
    feed(fsm, IN_ROI1, 0.6, clock)
    assert fsm.cycle_count == 1 and fsm.armed is False
    feed(fsm, IN_ROI1, 3.0, clock)          # 手留在最後一步的框裡
    assert fsm.alert_active is False
    assert fsm.current_step_idx == 0
    assert fsm.armed is False
    feed(fsm, NO_HAND, 0.6, clock)          # 離開所有 ROI 超過 cycle_arm_clear_sec
    assert fsm.armed is True
    assert len(_events(fsm, "armed")) == 1


def test_skip_alert(project, make_fsm, clock):
    fsm = make_fsm(project)
    feed(fsm, IN_ROI1, 1.1, clock)
    assert fsm.alert_active is True
    assert fsm.alert_message == "遺漏步驟：拿取標籤"
    assert len(_events(fsm, "alert_skip")) == 1


def test_skip_not_fired_on_pass_through(project, make_fsm, clock):
    fsm = make_fsm(project)
    feed(fsm, IN_ROI1, 0.5, clock)
    feed(fsm, NO_HAND, 0.5, clock)
    assert fsm.alert_active is False
    assert not _events(fsm, "alert_skip")


def test_timeout_alert(project, make_fsm, clock):
    fsm = make_fsm(project)
    feed(fsm, IN_ROI0, 0.5, clock)
    assert fsm.current_step_idx == 1
    feed(fsm, NO_HAND, 8.1, clock)
    assert fsm.alert_active is True
    assert fsm.alert_message == "步驟超時：貼合於產品表面"
    assert len(_events(fsm, "alert_timeout")) == 1


def test_alert_freezes_then_reset(project, make_fsm, clock):
    fsm = make_fsm(project)
    feed(fsm, IN_ROI1, 1.1, clock)
    assert fsm.alert_active is True
    feed(fsm, IN_ROI0, 1.0, clock)
    assert fsm.alert_active is True
    assert fsm.current_step_idx == 0
    assert not _events(fsm, "step_done")
    fsm.reset()
    assert fsm.alert_active is False
    assert fsm.alert_message == ""
    assert fsm.cycle_count == 0
    assert len(_events(fsm, "reset")) == 1


def test_dwell_resets_after_grace(project, make_fsm, clock):
    fsm = make_fsm(project)
    feed(fsm, IN_ROI0, 0.3, clock)
    feed(fsm, NO_HAND, 0.3, clock)
    feed(fsm, IN_ROI0, 0.3, clock)
    assert fsm.current_step_idx == 0
    assert not _events(fsm, "step_done")


def test_dwell_survives_short_miss(project, make_fsm, clock):
    fsm = make_fsm(project)
    feed(fsm, IN_ROI0, 0.3, clock)
    feed(fsm, NO_HAND, 0.1, clock)
    feed(fsm, IN_ROI0, 0.3, clock)
    assert fsm.current_step_idx == 1
    assert len(_events(fsm, "step_done")) == 1


@pytest.mark.parametrize("similarity,expect_done", [(0.9, True), (0.5, False)])
def test_visual_mode_pass_and_fail(project, make_fsm, clock, monkeypatch, similarity, expect_done):
    monkeypatch.setattr(core_detector, "roi_similarity", lambda a, b: similarity)
    fsm = make_fsm(project)
    feed(fsm, IN_ROI0, 0.5, clock)
    assert fsm.current_step_idx == 1
    feed(fsm, IN_ROI1, 0.6, clock)
    assert fsm.phase == "waiting_leave"
    feed(fsm, NO_HAND, 0.4, clock)
    assert fsm.last_similarity == similarity
    if expect_done:
        assert fsm.cycle_count == 1 and fsm.current_step_idx == 0
    else:
        assert fsm.cycle_count == 0 and fsm.current_step_idx == 1 and fsm.phase == "waiting_leave"


def test_reset_zone_clears_alert(project, make_fsm, clock):
    project.reset_zone = list(RESET_ZONE)
    fsm = make_fsm(project)
    feed(fsm, IN_ROI1, 1.1, clock)
    assert fsm.alert_active is True
    feed(fsm, IN_RESET, 1.1, clock)
    assert fsm.alert_active is False
    assert fsm.current_step_idx == 0
    assert fsm.armed is False
    assert len(_events(fsm, "reset")) == 1


def test_config_validation_overlap(tmp_path):
    cfg = sample_config()
    cfg["steps"][1]["roi"] = [200, 300, 200, 180]   # 與步驟 0 的 [100, 200, 150, 150] 重疊
    path = write_project(str(tmp_path / "bad"), cfg)
    with pytest.raises(ProjectConfigError) as exc:
        load_project(path)
    assert "重疊" in str(exc.value)


def test_config_validation_reset_zone_overlap(tmp_path):
    cfg = sample_config(reset_zone=[120, 220, 50, 50])
    path = write_project(str(tmp_path / "bad2"), cfg)
    with pytest.raises(ProjectConfigError):
        load_project(path)


def test_events_csv_written(project, clock, tmp_path):
    log_dir = str(tmp_path / "logs")
    fsm = core_detector.SOPStateMachine(project, error_log_dir=log_dir, sound_enabled=False)
    feed(fsm, IN_ROI1, 1.1, clock)
    csv_path = os.path.join(log_dir, "events.csv")
    assert os.path.isfile(csv_path)
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    assert lines[0] == "time_iso,event_type,step_index,step_name,message,screenshot"
    assert "alert_skip" in lines[1] and "遺漏步驟：拿取標籤" in lines[1]
    shots = [n for n in os.listdir(log_dir) if n.endswith("_skip_0.jpg")]
    assert len(shots) == 1
