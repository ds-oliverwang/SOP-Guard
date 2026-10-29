"""盒內狀態辨識（用灰階縮圖描述子、合成影像，不需要 CLIP）與 box_state 步驟。"""
from __future__ import annotations

import os

import numpy as np
import pytest

from core_detector import BoxStateRunner, FrameContext, SOPStateMachine, state_region
from project_store import project_from_dict
from state_recognizer import Embedder, StateLibrary, StateStream


def synth(kind: str, seed: int = 0):
    """合成盒內影像：空盒＝均勻紙板；有卡片＝左上一塊亮方塊；有鞋子＝中間大塊暗區＋紋理。"""
    rng = np.random.RandomState(seed)
    img = np.full((120, 200, 3), (120, 150, 180), np.uint8)
    img = np.clip(img.astype(int) + rng.normal(0, 6, img.shape).astype(int), 0, 255).astype(np.uint8)
    if kind == "card":
        img[10:40, 10:60] = (240, 240, 240)
    if kind == "shoe":
        img[20:100, 30:170] = (30, 30, 30)
        img[40:80, 60:140] = (90, 90, 90)
    return img


@pytest.fixture
def lib(tmp_path):
    lib = StateLibrary(str(tmp_path), embedder=Embedder(prefer_clip=False))
    for i in range(4):
        lib.add_example("空盒", synth("empty", i), frame=i)
        lib.add_example("已放卡片", synth("card", 10 + i), frame=10 + i)
        lib.add_example("已放鞋子", synth("shoe", 20 + i), frame=20 + i)
    return lib


def test_library_persists_and_recognizes(lib, tmp_path):
    assert lib.order == ["空盒", "已放卡片", "已放鞋子"] and lib.counts() == {"空盒": 4, "已放卡片": 4, "已放鞋子": 4}
    assert os.path.isfile(os.path.join(str(tmp_path), "states", "states.json"))
    lib2 = StateLibrary(str(tmp_path), embedder=Embedder(prefer_clip=False))   # 重新載入
    assert lib2.recognize(synth("card", 99)).state == "已放卡片"
    assert lib2.recognize(synth("shoe", 98)).state == "已放鞋子"
    assert lib2.recognize(synth("empty", 97)).state == "空盒"
    loo = lib2.leave_one_out()
    assert loo["n"] == 12 and loo["accuracy"] >= 0.9


def test_stream_votes_and_skips_when_hands_inside(lib):
    st = StateStream(lib, votes=5, min_votes=3)
    for i in range(3):
        st.push(synth("empty", 50 + i))
    assert st.current == "空盒"
    st.push(synth("shoe", 60), hands_inside=True)    # 手在盒內 → 不判，維持
    assert st.current == "空盒" and "手在盒內" in st.last.reason
    for i in range(5):
        st.push(synth("shoe", 70 + i))
    assert st.current == "已放鞋子"


def test_box_state_steps_in_checklist(tmp_path, lib):
    cfg = {
        "schema_version": 2, "project_name": "s", "video_file": "demo.avi", "resolution": [1280, 720], "fps": 30.0,
        "anchors": [{"name": "box", "prompt": "cardboard box"}], "zones": [],
        "state_anchor": "box", "state_every_n_frames": 1,
        "cycle_end": {"type": "box_state", "label": "空盒", "hold_sec": 1.0}, "cycle_min_sec": 0.0,
        "steps": [{"step_index": 0, "step_name": "放卡片", "type": "box_state", "label": "已放卡片", "dwell_time_sec": 0.3, "timeout_sec": 60},
                  {"step_index": 1, "step_name": "放鞋子", "type": "box_state", "label": "已放鞋子", "dwell_time_sec": 0.3, "timeout_sec": 60}],
    }
    p = project_from_dict(cfg, str(tmp_path))
    fsm = SOPStateMachine(p, sound_enabled=False); t = 0.0

    def feed(state, sec):
        nonlocal t
        for _ in range(int(sec * 30)):
            fsm.update(None, FrameContext(anchors={"box": [(0, 0, 200, 120)]}, box_state=state), t); t += 1 / 30
    feed("空盒", 0.5)            # 初始狀態不算步驟，也不是「一輪結束」（還沒開始）
    feed("已放卡片", 0.5)
    assert 0 in fsm.observed
    feed("已放鞋子", 0.5)
    assert 1 in fsm.observed
    feed("空盒", 1.2)            # 盒子回到空盒（或換下一個盒）＝一輪結束
    assert fsm.cycle_count == 1 and fsm.bad_cycle_count == 0


def test_runner_uses_anchor_region_and_hand_gate(tmp_path, lib):
    cfg = {"schema_version": 2, "project_name": "r", "video_file": "demo.avi", "resolution": [400, 240], "fps": 30.0,
           "anchors": [{"name": "box", "prompt": "cardboard box"}], "zones": [], "state_anchor": "box", "state_every_n_frames": 1,
           "steps": [{"step_index": 0, "step_name": "放卡片", "type": "box_state", "label": "已放卡片", "dwell_time_sec": 0.3, "timeout_sec": 60}]}
    p = project_from_dict(cfg, str(tmp_path))
    runner = BoxStateRunner(p)
    runner.lib.embedder = Embedder(prefer_clip=False)
    frame = np.zeros((240, 400, 3), np.uint8); frame[60:180, 100:300] = synth("card", 5)
    anchors = {"box": [(100, 60, 200, 120)]}
    for _ in range(runner._settle + 3):          # 盒子剛出現先等 0.5 秒（settle）再判，之後 3 次投票
        runner.update(frame, anchors, hand_points=[])
    assert runner.stream.current == "已放卡片" and runner.region == (100, 60, 200, 120)
    # 手只扶著盒緣（同一隻手只有 1 點在內縮區）→ 照判；手伸進盒內（≥2 點在內縮區）→ 不判
    runner.update(frame, anchors, hand_points=[(102.0, 62.0), (5.0, 5.0), (5.0, 5.0), (5.0, 5.0)])
    assert "手在盒內" not in runner.stream.last.reason
    runner.update(frame, anchors, hand_points=[(150.0, 100.0), (160.0, 110.0), (5.0, 5.0), (5.0, 5.0)])
    assert "手在盒內" in runner.stream.last.reason
    # 錨點只是沿用（超過 1 秒沒真的偵測到，可能已對著空桌面）→ 不判、狀態維持
    assert runner.update(frame, anchors, hand_points=[], anchor_age=2.0) == "已放卡片"
    assert "沒偵測到" in runner.stream.last.reason
    # 盒子不見 → 狀態清成不知道（上一輪留下的「蓋上」不能被下一輪當結束條件）
    assert runner.update(frame, {}, hand_points=[]) is None and runner.stream.current is None
    # 盒子換到新位置（被推走／剛放上來）→ 0.5 秒內不判；偵測在「整盒」「盒身」間切換（一框包住另一框）不算移動
    for _ in range(runner._settle + 3):          # 盒子重新出現也算剛換位置
        runner.update(frame, anchors, hand_points=[])
    assert runner.stream.current == "已放卡片"
    moved = {"box": [(100, 160, 200, 120)]}                       # 往下移了一整個盒高
    assert runner.update(frame, moved, hand_points=[]) == "已放卡片" and "移動中" in runner.stream.last.reason
    for _ in range(runner._settle):
        runner.update(frame, moved, hand_points=[])
    assert runner.debug["why"] == "judged"
    whole = {"box": [(100, 100, 200, 180), (100, 160, 200, 120)]}   # 整盒包住盒身 → 取盒身、區域不變、不算移動
    runner.update(frame, whole, hand_points=[])
    assert runner.region == (100, 160, 200, 120) and runner.debug["why"] == "judged"


def test_state_region_picks_innermost_box(tmp_path):
    cfg = {"schema_version": 2, "project_name": "g", "video_file": "demo.avi", "resolution": [400, 240], "fps": 30.0,
           "anchors": [{"name": "box", "prompt": "cardboard box", "max_count": 2}], "zones": [], "state_anchor": "box", "steps": []}
    p = project_from_dict(cfg, str(tmp_path))
    whole, body = (100, 40, 200, 160), (100, 100, 200, 100)     # 打開的盒子：整盒（含盒蓋）包住盒身 → 取盒身
    assert state_region(p, {"box": [whole, body]}) == body
    assert state_region(p, {"box": [body, whole]}) == body
    lid = (100, 0, 200, 90)                                      # 盒蓋、盒身分開不互包 → 照錨點順序取第一個
    assert state_region(p, {"box": [body, lid]}) == body
    assert state_region(p, {"box": []}) is None
