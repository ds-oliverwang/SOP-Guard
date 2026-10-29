"""介面冒煙測試（離屏模式，不開視窗、不用相機）：建專案 → 放一支合成示範影片 → 拉時間軸標兩段 → 存檔與截圖。"""
from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

import main_ui  # noqa: E402
import project_store  # noqa: E402
import training  # noqa: E402
from video_recorder import Recorder  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    return app


@pytest.fixture
def projects_root(tmp_path, monkeypatch):
    root = tmp_path / "sop_projects"; root.mkdir()
    monkeypatch.setattr(project_store, "projects_dir", lambda: str(root))
    monkeypatch.setattr(main_ui, "projects_dir", lambda: str(root))
    return root


def _write_demo(path: str, frames: int = 120, w: int = 320, h: int = 240) -> None:
    rec = Recorder(); rec.start(path, (w, h), 30.0)
    for i in range(frames):
        fr = np.zeros((h, w, 3), np.uint8); fr[:, :, 1] = (i * 2) % 255
        rec.write(fr)
    rec.stop()


@pytest.fixture(autouse=True)
def no_modal_dialogs(monkeypatch):
    """離屏模式下對話框沒人按確定會永遠卡住：把它們換成記錄訊息的假函式。"""
    shown = []
    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(main_ui.QMessageBox, name, staticmethod(lambda *a, **k: shown.append(a[2] if len(a) > 2 else "")))
    monkeypatch.setattr(main_ui.QMessageBox, "question", staticmethod(lambda *a, **k: main_ui.QMessageBox.Yes))
    return shown


def test_ui_create_project_label_segments_and_screenshots(qapp, projects_root, tmp_path, no_modal_dialogs, monkeypatch):
    d = main_ui.create_project_with_defaults("桌面測試", 320, 240, 30.0)
    assert os.path.isfile(os.path.join(d, "config.json"))
    _write_demo(os.path.join(d, "demo.avi"))

    win = main_ui.MainWindow()
    win.resize(1360, 860)
    assert win.project_list.count() == 1
    win.load_project("桌面測試")
    assert win.project is not None and win.reader is not None and win.reader.frame_count == 120

    # 分段：第 10～50 格「舉手」、第 60～110 格「壓下」
    win.tabs.setCurrentIndex(1)
    win.slider.setValue(10); win.set_start()
    assert "起點 ✓ 第 10 格" in win.btn_start.text() and win.timeline.start == 10
    win.slider.setValue(50); win.set_end()
    assert "終點 ✓ 第 50 格" in win.btn_end.text() and win.timeline.end == 50
    win.action_combo.setEditText("舉手"); win.add_segment()
    assert win.btn_start.text().startswith("設為起點") and win.timeline.start is None
    assert len(win.timeline.segments) == 1 and win.timeline.total == 120
    win.slider.setValue(60); win.set_start(); win.slider.setValue(110); win.set_end()
    win.action_combo.setEditText("壓下"); win.add_segment()
    assert [s["label"] for s in win.segments] == ["舉手", "壓下"]
    assert win.seg_table.rowCount() == 2
    saved = training.load_segments(win.project)
    assert saved == [{"label": "舉手", "start": 10, "end": 51}, {"label": "壓下", "start": 60, "end": 111}]

    # 太短的分段被擋
    win.slider.setValue(112); win.set_start(); win.slider.setValue(118); win.set_end()
    win.action_combo.setEditText("短"); win.add_segment()
    assert len(win.segments) == 2
    assert any("太短" in m or "不到 1 秒" in m for m in no_modal_dialogs)

    # 刪除第一段
    win.seg_table.selectRow(0); win.delete_segment()
    assert [s["label"] for s in win.segments] == ["壓下"]

    # 自動找其他輪：假的建議兩段 → 表格多兩列「建議」、時間軸有淡色塊 → 接受全部 → 變成已標
    monkeypatch.setattr(training, "propose_for_project",
                        lambda d, hm, yw, progress=None: [{"label": "壓下", "start": 10, "end": 45, "prob": 0.8},
                                                          {"label": "舉手", "start": 112, "end": 120, "prob": 0.6}])
    win.propose()
    deadline = time.time() + 5.0
    while time.time() < deadline and (win.propose_worker is None or win.propose_worker.isRunning() or not win.proposals):
        qapp.processEvents(); time.sleep(0.02)
    assert len(win.proposals) == 2 and win.seg_table.rowCount() == 3 and len(win.timeline.proposals) == 2
    assert "建議" in win.seg_table.item(1, 4).text()
    win.accept_all_proposals()
    assert len(win.proposals) == 0 and [s["label"] for s in win.segments] == ["壓下", "壓下", "舉手"]
    assert training.load_segments(win.project)[-1] == {"label": "舉手", "start": 112, "end": 120}

    # 盒內狀態：用假的盒子區域（不跑 YOLO），加兩個狀態各 3 格 → 產生步驟
    monkeypatch.setattr(win, "_detect_box_region", lambda frame: (40, 30, 200, 150))
    for st, frames_ in (("空盒", (5, 20, 35)), ("已放鞋子", (70, 90, 110))):
        for n in frames_:
            win.slider.setValue(n); win.state_combo.setEditText(st); win.add_state_example()
    lib = win._state_lib()
    assert lib.order == ["空盒", "已放鞋子"] and lib.counts() == {"空盒": 3, "已放鞋子": 3}
    assert win.state_list.count() == 2
    win.generate_state_steps()
    p2 = win.project
    assert [s.step_name for s in p2.steps if s.type == "box_state"] == ["已放鞋子"]
    assert p2.cycle_end.type == "object_absent" and p2.cycle_end.object == "box"   # 沒有「蓋上／結束」狀態 → 盒子拿走算一輪結束
    assert any(a.name == "box" for a in p2.anchors) and p2.state_anchor == "box"

    # 物件版：標空盒參考、框選物品（模擬拖拉結果）、清單；沒模型時「產生步驟」要擋下
    win.slider.setValue(5); win.mark_empty_ref()
    win.slider.setValue(40); win._sel_rect = (40, 30, 120, 80); win.item_combo.setEditText("標籤"); win.add_item_seed()
    win.slider.setValue(60); win._sel_rect = (30, 20, 200, 150); win.item_combo.setEditText("鞋子"); win.add_item_seed()
    win.slider.setValue(90); win.mark_closed_ref()
    seeds = win._item_seeds()
    assert seeds.refs == [5] and seeds.closed == [90] and seeds.order == ["標籤", "鞋子"] and seeds.counts() == {"標籤": 1, "鞋子": 1}
    assert win.item_list.count() == 3 and win._sel_rect is None
    win.generate_item_steps()
    assert [s.type for s in win.project.steps if s.type == "item"] == []        # 沒有模型 → 只提示，不產生

    shots = tmp_path / "shots"; shots.mkdir()
    for i, name in enumerate(("record", "label", "monitor")):
        win.tabs.setCurrentIndex(i)
        qapp.processEvents()
        assert win.grab().save(str(shots / f"ui_{name}.png"))
    win.stop_all(); win.close()
    assert all((shots / f"ui_{n}.png").stat().st_size > 1000 for n in ("record", "label", "monitor"))
    # 把截圖留一份到 cache 方便人工檢視
    cache = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "ui_shots")
    os.makedirs(cache, exist_ok=True)
    for n in ("record", "label", "monitor"):
        (shots / f"ui_{n}.png").replace(os.path.join(cache, f"ui_{n}.png"))
