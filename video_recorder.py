"""影片來源與影格讀取：相機來源（實際解析度、實測 fps）、MJPG 錄影、匯入既有影片、開檔轉正、影格讀取器、含中文路徑的圖片讀寫。"""
from __future__ import annotations

import os
import threading
import shutil
import time

import cv2
import numpy as np


def read_image(path: str):
    """cv2.imread 不支援含中文的路徑，改用 fromfile + imdecode。"""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: str, img) -> None:
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, 92] if ext.lower() in (".jpg", ".jpeg") else [])
    if not ok:
        raise IOError(f"影像編碼失敗：{path}")
    buf.tofile(path)


def open_video_oriented(path: str):
    """開啟影片並判斷旋轉標記。

    回傳 (cap, rotate_code 或 None, width, height, fps)，width/height 是「轉正後」的尺寸。
    手機錄的 mp4 常是橫向存檔 + 旋轉 90/270 標記；OpenCV 4.12 會自動轉正、4.11 不會，
    所以這裡用「標記說要轉、但讀到的影格仍是橫的」來判斷要不要自己轉。
    """
    if not os.path.isfile(path):
        raise IOError(f"找不到影片檔：{path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"無法開啟影片：{path}")
    meta = int(round(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)) % 360
    auto = cap.get(cv2.CAP_PROP_ORIENTATION_AUTO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise IOError(f"影片讀不到任何影格：{path}")
    h, w = frame.shape[:2]
    rotate = None
    if meta in (90, 270) and w > h:
        rotate = cv2.ROTATE_90_CLOCKWISE if meta == 90 else cv2.ROTATE_90_COUNTERCLOCKWISE
        w, h = h, w
    elif meta == 180 and not auto:
        rotate = cv2.ROTATE_180
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cap, rotate, w, h, fps


def is_network_source(source) -> bool:
    """rtsp:// http:// https:// 開頭的字串＝網路攝影機（IP cam）。"""
    return isinstance(source, str) and source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://"))


def is_camera_source(source) -> bool:
    """相機來源：整數編號（或純數字字串）或網路串流網址；其餘視為影片檔路徑。"""
    if isinstance(source, int):
        return True
    return isinstance(source, str) and (source.strip().isdigit() or is_network_source(source))


class CameraSource:
    """相機來源：USB 相機（編號）或 IP cam（rtsp:// 網址）。
    要求的解析度不一定拿得到，以讀回的實際值為準；相機回報 fps 為 0 時用前 measure_frames 格實測。
    IP cam 走 FFmpeg、RTSP 強制 TCP（UDP 在辦公室網路常掉包成花屏）、解碼器不緩衝；
    另開一條執行緒一直收、只留最新一格：處理跟不上相機 fps 時舊格直接丟，延遲才不會越積越多（實測處理約 10～17 fps、相機 30 fps，
    不丟格幾分鐘後會落後好幾秒）。dropped 可查丟了幾格。"""

    def __init__(self, cam_id=0, width: int = 1280, height: int = 720, measure_frames: int = 20, latest_only: bool = True):
        self.cam_id = cam_id
        self.is_network = is_network_source(cam_id)
        if self.is_network:
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000|fflags;nobuffer|flags;low_delay|max_delay;500000")
            self._cap = cv2.VideoCapture(str(cam_id), cv2.CAP_FFMPEG)
        else:
            idx = int(str(cam_id).strip())
            self._cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(idx)
        if not self._cap.isOpened():
            raise IOError(f"無法開啟相機 {cam_id}" + ("（請確認網址、帳號密碼與網路）" if self.is_network else ""))
        if self.is_network:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._cap.release()
            raise IOError(f"相機 {cam_id} 開了但讀不到影格")
        self.resolution = (frame.shape[1], frame.shape[0])
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 1.0:
            t0 = time.perf_counter(); n = 0
            for _ in range(measure_frames):
                ok, _f = self._cap.read()
                n += int(ok)
            el = time.perf_counter() - t0
            fps = n / el if el > 0 and n > 0 else 30.0
        self.fps = float(fps)
        self.frame_index = -1
        self.dropped = 0
        self._t0 = time.perf_counter()
        self._latest = None; self._seq = 0; self._taken = 0; self._err = False
        self._lock = threading.Lock(); self._new = threading.Event(); self._stop = threading.Event()
        self._thread = None
        if latest_only:
            self._thread = threading.Thread(target=self._grab_loop, daemon=True); self._thread.start()

    def _grab_loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._err = True; self._new.set(); return
            with self._lock:
                if self._seq > self._taken:
                    self.dropped += 1          # 上一格還沒被拿走就被新的蓋掉
                self._latest = frame; self._seq += 1
            self._new.set()

    def read(self):
        if self._thread is None:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                return None
            self.frame_index += 1
            return frame
        if not self._new.wait(timeout=5.0):
            return None
        with self._lock:
            if self._err and self._seq == self._taken:
                return None
            frame = self._latest; self._taken = self._seq; self._new.clear()
        self.frame_index += 1
        return frame

    def latest(self):
        """不等待、不算「拿走」：回傳 (序號, 最新一格)。給顯示執行緒用——畫面每格都要送，辨識執行緒照自己的速度 read()。
        還沒有影格時回 (0, None)。"""
        if self._thread is None:
            return 0, None
        with self._lock:
            return self._seq, self._latest

    @property
    def time_sec(self) -> float:
        return time.perf_counter() - self._t0

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._cap.release()


class Recorder:
    """MJPG 編碼寫成 .avi：每格都是關鍵格，之後用影格編號跳轉才準（mp4v 跳轉常常不準）。"""

    def __init__(self):
        self._writer = None
        self.frame_count = 0
        self.path = ""

    def start(self, path: str, resolution, fps: float) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), float(fps), (int(resolution[0]), int(resolution[1])))
        if not self._writer.isOpened():
            raise IOError(f"無法建立錄影檔：{path}")
        self.path = path
        self.frame_count = 0

    def write(self, frame) -> None:
        if self._writer is None:
            raise RuntimeError("Recorder 尚未 start")
        self._writer.write(frame)
        self.frame_count += 1

    def stop(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    @property
    def recording(self) -> bool:
        return self._writer is not None


def import_demo_video(src_path: str, project_dir: str) -> tuple[str, int, int, float]:
    """把影片原檔複製進專案資料夾（demo + 原副檔名），不重新編碼。回傳 (檔名, 轉正後寬, 轉正後高, fps)。"""
    if not os.path.isfile(src_path):
        raise IOError(f"找不到影片：{src_path}")
    cap, _rot, w, h, fps = open_video_oriented(src_path)
    cap.release()
    name = "demo" + (os.path.splitext(src_path)[1].lower() or ".mp4")
    dst = os.path.join(project_dir, name)
    if os.path.abspath(src_path) != os.path.abspath(dst):
        shutil.copy2(src_path, dst)
    return name, w, h, fps


class VideoFileSource:
    """影片檔當作畫面來源；介面與 CameraSource 相同。"""

    def __init__(self, path: str):
        self.path = path
        self._cap, self._rotate, w, h, self.fps = open_video_oriented(path)
        self.resolution = (w, h)
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.frame_index = -1

    def read(self):
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        self.frame_index += 1
        if self._rotate is not None:
            frame = cv2.rotate(frame, self._rotate)
        return frame

    @property
    def time_sec(self) -> float:
        return self.frame_index / self.fps if self.fps else 0.0

    def release(self) -> None:
        self._cap.release()


class FrameReader:
    """依影格編號取單格（標註用）。set 後驗證位置，不準就從頭循序讀。"""

    def __init__(self, path: str):
        self.path = path
        self._cap, self._rotate, w, h, self.fps = open_video_oriented(path)
        self.resolution = (w, h)
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._cache_index = None
        self._cache_frame = None

    def get_frame(self, n: int):
        if self._cache_index == n and self._cache_frame is not None:
            return self._cache_frame
        n = max(0, n)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, frame = self._cap.read()
        if not ok or int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) != n + 1:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame = None
            for _ in range(n + 1):
                ok, frame = self._cap.read()
                if not ok:
                    break
            if frame is None:
                raise IOError(f"讀不到第 {n} 格：{self.path}")
        if self._rotate is not None:
            frame = cv2.rotate(frame, self._rotate)
        self._cache_index, self._cache_frame = n, frame
        return frame

    def release(self) -> None:
        self._cap.release()
