"""網頁版的每輪紀錄（CycleTracker）：用假的狀態機模擬，不載模型。"""
from __future__ import annotations

from types import SimpleNamespace

from monitor_service import CycleTracker


def _steps():
    return [SimpleNamespace(step_index=0, step_name="藍盒", required=True),
            SimpleNamespace(step_index=1, step_name="貼標", required=True),
            SimpleNamespace(step_index=2, step_name="翻面", required=False)]


def _fsm(**kw):
    d = dict(mode="checklist", observed=set(), acc_sec={}, cycle_active=False, cycle_started_at=0.0, cycle_count=0,
             bad_cycle_count=0, alert_active=False, alert_type="", current_step_idx=0, dwell_sec=0.0)
    d.update(kw)
    return SimpleNamespace(**d)


def test_good_cycle_records_duration_and_step_times():
    tr = CycleTracker(_steps()); f = _fsm()
    tr.update(f, 0.0, True)
    f.cycle_active = True; f.cycle_started_at = 1.0; f.acc_sec = {0: 0.2}
    tr.update(f, 1.2, True)
    assert tr.info(1.2)["doing"] == [0] and tr.info(1.2)["elapsed"] == 0.2
    f.observed = {0}; f.acc_sec = {0: 0.5}
    tr.update(f, 3.0, True)
    assert tr.info(3.0)["done_at"] == {"0": 2.0} and tr.info(3.0)["doing"] == []
    f.observed = {0, 1}; f.acc_sec = {0: 0.5, 1: 0.5}
    tr.update(f, 6.0, True)
    f.cycle_active = False; f.cycle_count = 1; f.observed = set(); f.acc_sec = {}
    tr.update(f, 11.0, False)
    info = tr.info(11.0)
    assert info["history"][0]["ok"] and info["history"][0]["dur"] == 10.0 and info["elapsed"] is None
    assert info["avg_dur"] == 10.0 and info["top_missed"] == []


def test_bad_cycle_lists_missing_required_steps_only():
    tr = CycleTracker(_steps()); f = _fsm(cycle_active=True, cycle_started_at=0.0, observed={0})
    tr.update(f, 1.0, True)
    f.cycle_active = False; f.bad_cycle_count = 1; f.alert_active = True; f.alert_type = "alert_missing"; f.observed = set()
    tr.update(f, 8.0, True)
    info = tr.info(8.0)
    assert info["history"][0]["kind"] == "漏做" and info["history"][0]["missing"] == ["貼標"]
    assert info["top_missed"] == [{"name": "貼標", "count": 1}]


def test_discarded_cycle_is_not_recorded():
    tr = CycleTracker(_steps()); f = _fsm(cycle_active=True, cycle_started_at=0.0, acc_sec={0: 0.1})
    tr.update(f, 0.5, True)
    f.cycle_active = False; f.acc_sec = {}
    tr.update(f, 2.0, True)
    info = tr.info(2.0)
    assert info["history"] == [] and info["elapsed"] is None and info["doing"] == []


def test_doing_flag_expires_when_progress_stops():
    tr = CycleTracker(_steps()); f = _fsm(cycle_active=True, acc_sec={1: 0.1})
    tr.update(f, 1.0, True)
    assert tr.info(1.0)["doing"] == [1]
    tr.update(f, 1.5, True)
    assert tr.info(1.5)["doing"] == [1]
    tr.update(f, 2.0, True)
    assert tr.info(2.0)["doing"] == []


def test_hand_age():
    tr = CycleTracker(_steps()); f = _fsm()
    tr.update(f, 1.0, True); tr.update(f, 40.0, False)
    assert tr.info(40.0)["hand_age"] == 39.0


def test_first_update_only_sets_baseline():
    tr = CycleTracker(_steps()); f = _fsm(cycle_count=5, bad_cycle_count=1)
    tr.update(f, 1.0, True)
    assert tr.info(1.0)["history"] == []


def test_duration_stats_use_completed_cycles_only():
    tr = CycleTracker(_steps()); f = _fsm()
    tr.update(f, 0.0, True)
    for start, end, good in ((0.0, 30.0, True), (40.0, 60.0, False), (70.0, 110.0, True)):
        f.cycle_active = True; f.cycle_started_at = start; f.observed = {0}
        tr.update(f, start + 0.1, True)
        f.cycle_active = False; f.observed = set()
        if good:
            f.cycle_count += 1
        else:
            f.bad_cycle_count += 1
        tr.update(f, end, True)
    info = tr.info(110.0)
    assert info["avg_dur"] == 35.0
    assert info["dur_stats"] == {"min": 30.0, "max": 40.0, "last": 40.0, "n": 2}
    assert info["kind_count"]["漏做"] == 1
    assert info["miss_by_step"] == [{"name": "藍盒", "count": 0}, {"name": "貼標", "count": 1}]
