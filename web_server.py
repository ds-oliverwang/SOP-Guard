"""SOP-Guard 網頁版（第一階段：監控頁）。純標準函式庫（http.server），不用另外裝套件。

    python web_server.py [--port 8765] [--host 0.0.0.0]

- GET  /                 監控頁（web/index.html）
- GET  /stream.mjpg      疊好字的畫面串流（MJPEG；每張 JPEG 只編碼一次，多個瀏覽器共用）
- GET  /api/state        目前狀態（JSON：清單、輪數、警告、事件）
- GET  /api/projects     專案清單
- POST /api/start        {"project": 名稱, "source": "demo" | "camera"}
- POST /api/stop         停止
- POST /api/reset        解除警告／重置
- POST /api/screenshot   存一張截圖到 cache/
登入：同資料夾有 web_password.txt 時，瀏覽器要輸入密碼（帳號隨便、密碼＝檔案內容；HTTP Basic，區網用）。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from project_store import app_base_dir, list_projects, load_project  # noqa: E402

WEB_DIR = os.path.join(ROOT, "web")
PASSWORD_FILE = os.path.join(app_base_dir(), "web_password.txt")


def read_password() -> str | None:
    try:
        with open(PASSWORD_FILE, "r", encoding="utf-8-sig") as f:
            return f.read().strip() or None
    except OSError:
        return None


def project_rows() -> list[dict]:
    rows = []
    for name in list_projects():
        try:
            p = load_project(name)
            rows.append({"name": name, "steps": len(p.steps), "mode": p.mode if p.steps else None,
                         "has_demo": os.path.isfile(p.video_path), "ok": True})
        except Exception as e:
            rows.append({"name": name, "steps": 0, "mode": None, "has_demo": False, "ok": False, "error": str(e)})
    return rows


def parse_multipart(body: bytes, content_type: str) -> tuple[str, bytes] | None:
    """最簡單的 multipart/form-data：取第一個有 filename 的欄位，回傳 (檔名, 內容)。上傳影片用。"""
    if "boundary=" not in content_type:
        return None
    boundary = content_type.split("boundary=", 1)[1].strip().strip('"').encode()
    for part in body.split(b"--" + boundary):
        if b"filename=" not in part:
            continue
        head, _, data = part.partition(b"\r\n\r\n")
        fn = head.split(b"filename=", 1)[1].split(b"\r\n", 1)[0].strip().strip(b'"')
        if data.endswith(b"\r\n"):
            data = data[:-2]
        return fn.decode("utf-8", errors="replace"), data
    return None


class Handler(BaseHTTPRequestHandler):
    service = None          # MonitorService（或測試用的替身）
    label = None            # LabelService（標記頁；None＝只有監控頁）
    password: str | None = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):        # 串流每張都印會洗版
        msg = fmt % args
        if "stream.mjpg" in msg or "/api/state" in msg or "/api/train/state" in msg or "/api/record/state" in msg or "/api/frame" in msg:
            return
        super().log_message(fmt, *args)

    def _query(self) -> dict:
        q = self.path.split("?", 1)[1] if "?" in self.path else ""
        from urllib.parse import parse_qs
        return {k: v[0] for k, v in parse_qs(q).items()}

    # ---- 共用

    def _authed(self) -> bool:
        if not self.password:
            return True
        h = self.headers.get("Authorization", "")
        if h.startswith("Basic "):
            try:
                _, _, pw = base64.b64decode(h[6:]).decode("utf-8").partition(":")
                return pw == self.password
            except Exception:
                return False
        return False

    def _deny(self) -> None:
        body = "請輸入密碼".encode("utf-8")
        self.send_response(401); self.send_header("WWW-Authenticate", 'Basic realm="SOP-Guard"')
        self.send_header("Content-Type", "text/plain; charset=utf-8"); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def _file(self, path: str, ctype: str) -> None:
        try:
            with open(path, "rb") as f:
                body = f.read()
        except OSError:
            self._json({"error": "找不到頁面"}, 404); return
        self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def _read_json(self) -> dict | None:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if not raw:
            return {}
        try:
            d = json.loads(raw.decode("utf-8"))
            return d if isinstance(d, dict) else None
        except Exception:
            return None

    # ---- 路由

    def do_GET(self):
        if not self._authed():
            self._deny(); return
        path = self.path.split("?", 1)[0]
        try:
            if path in ("/", "/index.html"):
                self._file(os.path.join(WEB_DIR, "index.html"), "text/html; charset=utf-8")
            elif path in ("/label", "/label.html"):
                self._file(os.path.join(WEB_DIR, "label.html"), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._json(self.service.snapshot())
            elif path == "/api/projects":
                self._json({"projects": project_rows()})
            elif path == "/stream.mjpg":
                self._stream(self.service)
            elif self.label is not None and path == "/api/project":
                self._json(self.label.project_info(self._query().get("name", "")))
            elif self.label is not None and path == "/api/frame":
                q = self._query(); w = q.get("w")
                data = self.label.frame_jpeg(q.get("project", ""), int(q.get("n", 0) or 0), int(w) if w else None)
                if data is None:
                    self._json({"error": "這個專案沒有示範影片"}, 404); return
                self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
            elif self.label is not None and path == "/api/train/state":
                self._json(self.label.train.state())
            elif self.label is not None and path == "/api/record/state":
                self._json(self.label.record.state())
            elif self.label is not None and path == "/api/record/stream.mjpg":
                self._stream(self.label.record)
            else:
                self._json({"error": "沒有這個路徑"}, 404)
        except ValueError as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        if not self._authed():
            self._deny(); return
        path = self.path.split("?", 1)[0]
        if path == "/api/project/upload":
            self._upload(); return
        body = self._read_json()
        if body is None:
            self._json({"ok": False, "error": "內容不是 JSON"}, 400); return
        try:
            if path == "/api/start":
                self.service.start(str(body.get("project", "")), str(body.get("source", "demo")))
                self._json({"ok": True})
            elif path == "/api/stop":
                self.service.stop(); self._json({"ok": True})
            elif path == "/api/reset":
                self.service.request_reset(); self._json({"ok": True})
            elif path == "/api/screenshot":
                out = self.service.save_screenshot()
                self._json({"ok": out is not None, "path": out, "error": None if out else "還沒有監控畫面"})
            elif self.label is not None and path == "/api/seeds":
                self._json({"ok": True, "seeds": self.label.seed_action(str(body.get("project", "")), body)})
            elif self.label is not None and path == "/api/project/create":
                self._json({"ok": True, **self.label.create(str(body.get("name", "")))})
            elif self.label is not None and path == "/api/train/start":
                self.label.train.start(str(body.get("project", ""))); self._json({"ok": True})
            elif self.label is not None and path == "/api/steps/generate":
                self._json({"ok": True, **self.label.generate_steps(str(body.get("project", "")))})
            elif self.label is not None and path == "/api/record/start":
                self.label.record_start(str(body.get("project", "")), body.get("source")); self._json({"ok": True})
            elif self.label is not None and path == "/api/record/begin":
                self.label.record.begin_recording(); self._json({"ok": True})
            elif self.label is not None and path == "/api/record/stop":
                self._json({"ok": True, **self.label.record_stop()})
            else:
                self._json({"ok": False, "error": "沒有這個路徑"}, 404)
        except ValueError as e:
            self._json({"ok": False, "error": str(e)}, 400)
        except Exception as e:
            self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)

    def _upload(self):
        if self.label is None:
            self._json({"ok": False, "error": "沒有這個路徑"}, 404); return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > 2 * 1024 ** 3:
                raise ValueError("檔案大小不合理（上限 2 GB）")
            body = self.rfile.read(n)
            part = parse_multipart(body, self.headers.get("Content-Type", ""))
            if part is None:
                raise ValueError("沒有收到檔案")
            self._json({"ok": True, **self.label.import_upload(self._query().get("name", ""), part[0], part[1])})
        except ValueError as e:
            self._json({"ok": False, "error": str(e)}, 400)
        except Exception as e:
            self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)

    def _stream(self, source):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store"); self.send_header("Connection", "close"); self.end_headers()
        last = -1; idle = 0
        try:
            while True:
                fid, jpeg = source.wait_frame(last, timeout=1.0)
                if jpeg is None or fid == last:
                    idle += 1
                    if idle > 30 and not source.running:      # 停了 30 秒就收掉連線，瀏覽器會顯示最後一張
                        break
                    time.sleep(0.05)                          # 來源沒等就回同一張時不要空轉
                    continue
                idle = 0; last = fid
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n")
                self.wfile.write(jpeg); self.wfile.write(b"\r\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass


def make_server(service, host: str = "0.0.0.0", port: int = 8765, password: str | None = None, label=None) -> ThreadingHTTPServer:
    cls = type("BoundHandler", (Handler,), {"service": service, "password": password, "label": label})
    srv = ThreadingHTTPServer((host, port), cls)
    srv.daemon_threads = True
    return srv


def local_ips() -> list[str]:
    ips = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in ips and not ip.startswith("127."):
                ips.append(ip)
    except Exception:
        pass
    return ips


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SOP-Guard 網頁版監控")
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)
    from label_service import LabelService
    from monitor_service import MonitorService
    pw = read_password()
    srv = make_server(MonitorService(), a.host, a.port, pw, label=LabelService())
    print("SOP-Guard 網頁版已啟動" + ("（有密碼）" if pw else "（沒有密碼：同資料夾放 web_password.txt 就會要求輸入）"))
    print(f"  本機：http://127.0.0.1:{a.port}/   監控頁；標記頁 http://127.0.0.1:{a.port}/label")
    for ip in local_ips():
        print(f"  區網：http://{ip}:{a.port}/")
    print("按 Ctrl+C 結束")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            srv.RequestHandlerClass.service.stop()
        except Exception:
            pass
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
