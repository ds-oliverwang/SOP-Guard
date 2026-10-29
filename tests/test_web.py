"""網頁版：路由、JSON、串流、密碼。用替身服務，不載模型。"""
from __future__ import annotations

import base64
import json
import threading
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

import web_server


class _StubService:
    def __init__(self):
        self.running = False; self.started = []; self.resets = 0; self.stops = 0
        ok, buf = cv2.imencode(".jpg", np.full((24, 32, 3), 120, np.uint8)); self._jpeg = buf.tobytes(); self._fid = 1

    def snapshot(self):
        return {"running": self.running, "loading": None, "error": None, "finished": None, "project": "p", "source_kind": "demo",
                "frame_id": self._fid, "cycle_count": 2, "bad_cycle_count": 1, "alert_active": True, "alert_message": "漏做：標籤",
                "armed": True, "mode": "checklist", "steps": [{"index": 0, "name": "標籤", "state": "pending", "required": True, "acc": 0.2, "dwell": 0.5}],
                "events": [{"t": 1.0, "type": "alert_missing", "message": "漏做：標籤", "step_index": ""}], "frames": 10, "fps": 12.0, "n_events": 1}

    def start(self, project, source_kind="demo", source=None):
        if project == "bad":
            raise ValueError("這個專案還沒有步驟")
        self.running = True; self.started.append((project, source_kind))

    def stop(self, timeout=5.0):
        self.running = False; self.stops += 1

    def request_reset(self):
        self.resets += 1

    def latest_jpeg(self):
        return self._jpeg

    def wait_frame(self, last_id, timeout=1.0):
        return self._fid, self._jpeg

    def save_screenshot(self):
        return r"C:\x\shot.jpg"


@pytest.fixture
def server():
    svc = _StubService(); srv = web_server.make_server(svc, "127.0.0.1", 0)
    th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
    yield svc, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown(); srv.server_close()


def _get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, r.read()


def _post(url, body, headers=None):
    data = json.dumps(body).encode("utf-8") if body is not None else b"{"
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8")
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"raw": raw}          # 401 是純文字「請輸入密碼」


def test_page_state_and_actions(server):
    svc, base = server
    st, body = _get(base + "/"); assert st == 200 and "SOP-Guard" in body.decode("utf-8") and "啟動監控" in body.decode("utf-8")
    st, body = _get(base + "/api/state"); d = json.loads(body)
    assert st == 200 and d["alert_message"] == "漏做：標籤" and d["steps"][0]["name"] == "標籤"
    st, d = _post(base + "/api/start", {"project": "p1", "source": "camera"}); assert st == 200 and d["ok"] and svc.started == [("p1", "camera")]
    st, d = _post(base + "/api/start", {"project": "bad"}); assert st == 400 and "步驟" in d["error"]
    st, d = _post(base + "/api/reset", {}); assert d["ok"] and svc.resets == 1
    st, d = _post(base + "/api/stop", {}); assert d["ok"] and svc.stops == 1 and not svc.running
    st, d = _post(base + "/api/screenshot", {}); assert d["ok"] and d["path"].endswith("shot.jpg")
    st, d = _post(base + "/api/start", None); assert st == 400            # 不是 JSON
    st, body = _get(base + "/api/projects"); assert st == 200 and "projects" in json.loads(body)


def test_stream_sends_multipart_jpeg(server):
    svc, base = server
    req = urllib.request.Request(base + "/stream.mjpg")
    with urllib.request.urlopen(req, timeout=5) as r:
        assert r.headers["Content-Type"].startswith("multipart/x-mixed-replace")
        head = r.readline(); assert head.strip() == b"--frame"
        ctype = r.readline(); clen = r.readline(); r.readline()
        n = int(clen.split(b":")[1]); data = r.read(n)
        assert ctype.strip() == b"Content-Type: image/jpeg" and data == svc.latest_jpeg()


def test_password_required_when_set():
    svc = _StubService(); srv = web_server.make_server(svc, "127.0.0.1", 0, password="abc")
    th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        with pytest.raises(urllib.error.HTTPError) as ei:
            _get(base + "/api/state")
        assert ei.value.code == 401
        auth = {"Authorization": "Basic " + base64.b64encode(b"anyone:abc").decode()}
        st, body = _get(base + "/api/state", auth); assert st == 200
        st, d = _post(base + "/api/reset", {}, {"Authorization": "Basic " + base64.b64encode(b"x:wrong").decode()}); assert st == 401
    finally:
        srv.shutdown(); srv.server_close()


def _video_project(tmp_path, name="v"):
    """暫時的專案資料夾（在 tmp 下，不碰 sop_projects）＋ 60 格合成影片。回傳專案路徑。"""
    import json as _json
    d = tmp_path / name; d.mkdir()
    wr = cv2.VideoWriter(str(d / "demo.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (320, 240))
    for k in range(60):
        fr = np.full((240, 320, 3), 40, np.uint8); cv2.rectangle(fr, (100, 80), (180, 140), (200, 220, 230), -1); wr.write(fr)
    wr.release()
    cfg = {"schema_version": 2, "project_name": name, "video_file": "demo.avi", "resolution": [320, 240], "fps": 30.0, "anchors": [], "zones": [], "steps": []}
    with open(d / "config.json", "w", encoding="utf-8") as f:
        _json.dump(cfg, f)
    return str(d)


def test_label_service_frames_seeds_and_multipart(tmp_path):
    from label_service import LabelService
    pdir = _video_project(tmp_path)
    svc = LabelService()
    info = svc.project_info(pdir)
    assert info["has_video"] and info["frame_count"] == 60 and info["resolution"] == [320, 240] and info["seeds"]["order"] == []
    jpg = svc.frame_jpeg(pdir, 10); assert jpg[:2] == b"\xff\xd8"
    small = cv2.imdecode(np.frombuffer(svc.frame_jpeg(pdir, 10, max_w=160), np.uint8), cv2.IMREAD_COLOR); assert small.shape[1] == 160
    s = svc.seed_action(pdir, {"action": "add_ref", "frame": 3}); assert s["refs"] == [3]
    s = svc.seed_action(pdir, {"action": "add_seed", "frame": 10, "name": "標籤", "rect": [100, 80, 80, 60]}); assert s["counts"] == {"標籤": 1}
    with pytest.raises(ValueError):
        svc.seed_action(pdir, {"action": "add_seed", "frame": 10, "name": "x", "rect": [0, 0, 5, 5]})
    s = svc.seed_action(pdir, {"action": "set_region", "rect": [0, 0, 320, 240]}); assert s["region"] == [0, 0, 320, 240]
    s = svc.seed_action(pdir, {"action": "set_box_label", "name": "標籤", "is_box": True}); assert s["box_labels"] == ["標籤"]
    s = svc.seed_action(pdir, {"action": "add_hand_step", "name": "檢查", "prompt": "shoe", "dwell": 0.7}); assert s["hand_steps"][0]["prompt"] == "shoe"
    s = svc.seed_action(pdir, {"action": "remove_seed", "index": 0}); assert s["counts"] == {}
    s = svc.seed_action(pdir, {"action": "remove_ref", "frame": 3}); assert s["refs"] == []
    with pytest.raises(ValueError):
        svc.generate_steps(pdir)                       # 沒有物品
    body = (b"--XX\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.mp4\"\r\nContent-Type: video/mp4\r\n\r\nDATA123\r\n--XX--\r\n")
    assert web_server.parse_multipart(body, "multipart/form-data; boundary=XX") == ("a.mp4", b"DATA123")
    assert web_server.parse_multipart(b"x", "text/plain") is None


def test_label_service_create_and_upload(tmp_path, monkeypatch):
    import label_service
    monkeypatch.setattr(label_service, "projects_dir", lambda: str(tmp_path))
    svc = label_service.LabelService()
    d = svc.create("新案")
    assert d["name"] == "新案" and not svc.project_info(str(tmp_path / "新案"))["has_video"]
    with pytest.raises(Exception):
        svc.create("新案")                               # 重複
    src = _video_project(tmp_path, "src")
    with open(src + "/demo.avi", "rb") as f:
        data = f.read()
    r = svc.import_upload(str(tmp_path / "新案"), "clip.avi", data)
    assert r["video"] == "demo.avi" and r["resolution"] == [320, 240]
    assert svc.project_info(str(tmp_path / "新案"))["frame_count"] == 60
    with pytest.raises(ValueError):
        svc.import_upload(str(tmp_path / "新案"), "x.txt", b"nope")


def test_label_routes_with_stub(server):
    svc, base = server
    st, body = _get(base + "/label"); assert st == 200 and "標記物品" in body.decode("utf-8")
    st, d = _post(base + "/api/seeds", {"project": "p", "action": "add_ref", "frame": 1}); assert st == 404   # 沒掛 label 服務


def test_draw_texts_cached_blit_matches_bounds():
    """文字快取貼圖：畫得出中文、超出邊界會裁掉不炸、同一行字第二次不再呼叫 PIL。"""
    import core_detector as cd
    if cd._cjk_font(22) is None:
        pytest.skip("沒有中文字型")
    fr = np.zeros((120, 200, 3), np.uint8)
    out = cd.draw_texts(fr, [(10, 10, "清單模式 完成 1 輪", (255, 255, 255), 22), (150, 100, "超出邊界的字", (0, 255, 0), 22), (-5, -5, "負座標", (0, 0, 255), 20)])
    assert out is fr and fr.max() > 200 and fr[15:35, 10:60].max() > 0
    n = len(cd._text_cache); cd.draw_texts(np.zeros((120, 200, 3), np.uint8), [(10, 10, "清單模式 完成 1 輪", (255, 255, 255), 22)])
    assert len(cd._text_cache) == n


def test_monitor_service_snapshot_before_start():
    from monitor_service import MonitorService
    s = MonitorService(); d = s.snapshot()
    assert d["running"] is False and d["project"] is None and s.latest_jpeg() is None and s.save_screenshot() is None
    with pytest.raises(ValueError):
        s.start("沒有這個專案_xyz")
