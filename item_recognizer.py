"""物品辨識（物件版）：操作者在示範影片上「框一次、取名」，程式把它延伸成訓練資料，本機微調一個小偵測模型（yolov8n），
監控時在盒內找每一樣物品。

流程：
  1. 追蹤（trace）：整支示範影片跑一次 YOLO 找盒子（每 15 格）＋手部（每格），存 <專案>/items/trace.json。
  2. 操作者標：某一格「這是空盒」（每次盒子放好後各一次）；在畫面上框物品、取名（seeds.json）。
  3. 延伸：同一次擺放（盒子沒移動）裡，手不在盒內的格，用樣板比對找到同一物品就一起標（物品在空盒參考上也配得到的不算，
     大塊深色的鞋子會配到陰影）。寫成 YOLO 格式資料集 <專案>/items/dataset/。
  4. 訓練：tools/train_items.py（子程序），模型存 <專案>/items/model/best.pt。
  5. 監控：BoxStateRunner 在盒內裁圖上跑模型 → 哪些物品在；步驟型別 item。
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field

import cv2
import numpy as np

ITEMS_DIR = "items"
SEEDS_FILE = os.path.join(ITEMS_DIR, "seeds.json")
TRACE_FILE = os.path.join(ITEMS_DIR, "trace.json")
MODEL_DIR = os.path.join(ITEMS_DIR, "model")
MODEL_FILE = os.path.join(MODEL_DIR, "best.pt")


def dataset_dir(project) -> str:
    """訓練資料集放在純 ASCII 路徑：ultralytics／cv2.imread 在 Windows 吃不了中文路徑（專案名常是中文）。"""
    import hashlib
    root = os.path.dirname(os.path.abspath(__file__))
    key = hashlib.md5(os.path.abspath(project.dir_path).encode("utf-8")).hexdigest()[:10]
    return os.path.join(root, "cache", "item_ds_" + key)


def train_dir(project) -> str:
    return dataset_dir(project) + "_model"
TRACE_STEP = 5                  # 手部每幾格記一次（延伸與判定都用這個取樣）
DET_EVERY = 15                  # 盒子偵測每幾格一次
MATCH_MIN = 0.75                # 樣板比對最低分數
MATCH_OVER_EMPTY = 0.05         # 比「在空盒參考上」至少高這麼多才算真的有東西
MATCH_MIN_IOU = 0.3             # 比對到的位置要跟框選時的位置重疊（同一次擺放裡東西不會移動）；沒有這條，鞋子的樣板會配到陰影、手，
                                # 標出來的框漂到別處，模型學成一團（使用者桌面測試專案：鞋子 63 筆標籤、訓完 158 格只認到 1 格）
FIXED_WINDOW_SEC = 2.0          # 看整張桌面（固定範圍）時，種子只往前後這麼多秒延伸：現場一個動作只停 1～2 秒，東西就被翻面／推走，
                                # 整支影片比對既慢（1080×830 的桌面 × 上千格）又會配到別輪長得一樣的東西
FIXED_IMGSZ = 960               # 固定桌面範圍的裁圖比盒內大很多（約 1000 px 寬），640 會把 90 px 的貼紙縮到 50 px；訓練與偵測都用 960

Rect = tuple[int, int, int, int]


def is_fixed(project) -> bool:
    """物件版有兩種看法：跟著盒子裁（state_fixed 為 None）／看整張桌面的固定範圍（state_fixed 有值）。
    固定範圍時物品偵測不等盒子出現、不看錨點新不新鮮：翻鞋舌檢查發生在盒子放上來之前、鞋子放到盒上後盒子會被遮住。"""
    return getattr(project, "state_fixed", None) is not None


# ---------------------------------------------------------------- 種子（操作者的標記）

@dataclass
class Seed:
    name: str
    frame: int
    rect: list[int]             # 畫面座標 [x, y, w, h]
    auto: bool = False          # True＝框選後自動往前後追出來的框（使用者看得到、可以刪），訓練時直接當標籤、不再延伸
    group: int = 0              # 同一次框選（手框的那一個＋它追出來的）共用一個編號；刪手框的那一個會連自動的一起刪


class ItemSeeds:
    """<專案>/items/seeds.json：{"order": [物品名...], "refs": [空盒參考格...], "closed": [蓋著的盒子格...], "seeds": [{"name","frame","rect"}],
    "region": [x, y, w, h] 或 null, "box_labels": [物品名...]}
    refs 兼作「開著的盒子」範例、closed 是「蓋著的盒子」範例：蓋著的盒子 YOLO 一樣認成盒子，要靠這兩組分開（蓋著＝不算盒子在）。
    region 有值＝看整張桌面的固定範圍（refs 就是「空桌面」格）；box_labels＝哪些物品是「盒子的樣子」（立著、翻面、鞋子放在上面…），
    看到任一個就當盒子還在桌上，推走後才算一輪結束。"""

    def __init__(self, project_dir: str):
        self.path = os.path.join(project_dir, SEEDS_FILE)
        self.order: list[str] = []
        self.refs: list[int] = []
        self.closed: list[int] = []
        self.seeds: list[Seed] = []
        self.region: list[int] | None = None
        self.box_labels: list[str] = []
        self.hand_steps: list[dict] = []     # [{"name": 步驟名, "prompt": 物件的英文提示（YOLO-World）, "dwell": 連續停留秒數}]
        self.load()

    def load(self) -> None:
        self.order, self.refs, self.closed, self.seeds, self.region, self.box_labels, self.hand_steps = [], [], [], [], None, [], []
        if os.path.isfile(self.path):
            with open(self.path, "r", encoding="utf-8-sig") as f:
                d = json.load(f)
            self.order = [s for s in d.get("order", []) if isinstance(s, str)]
            self.refs = sorted({int(x) for x in d.get("refs", [])})
            self.closed = sorted({int(x) for x in d.get("closed", [])})
            self.seeds = [Seed(str(s["name"]), int(s["frame"]), [int(v) for v in s["rect"]], bool(s.get("auto", False)), int(s.get("group", 0) or 0))
                          for s in d.get("seeds", []) if isinstance(s, dict)]
            reg = d.get("region")
            self.region = [int(v) for v in reg] if isinstance(reg, list) and len(reg) == 4 else None
            self.box_labels = [s for s in d.get("box_labels", []) if isinstance(s, str) and s in self.order]
            self.hand_steps = [{"name": str(h["name"]), "prompt": str(h["prompt"]), "dwell": float(h.get("dwell", 0.7))}
                               for h in d.get("hand_steps", []) if isinstance(h, dict) and h.get("name") and h.get("prompt")]

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"order": self.order, "refs": self.refs, "closed": self.closed, "seeds": [s.__dict__ for s in self.seeds],
                       "region": self.region, "box_labels": self.box_labels, "hand_steps": self.hand_steps}, f, ensure_ascii=False, indent=2)

    def add_hand_step(self, name: str, prompt: str, dwell: float = 0.7) -> None:
        """「手停在某個物件上」的步驟：物件用 YOLO-World 的英文提示找（例如 shoe），不用框、不用訓練、換款式不用改。"""
        name = name.strip(); prompt = prompt.strip()
        self.hand_steps = [h for h in self.hand_steps if h["name"] != name] + [{"name": name, "prompt": prompt, "dwell": float(dwell)}]
        self.save()

    def remove_hand_step(self, name: str) -> None:
        self.hand_steps = [h for h in self.hand_steps if h["name"] != name]; self.save()

    def set_region(self, rect: Rect | None) -> None:
        self.region = None if rect is None else [int(v) for v in rect]; self.save()

    def set_box_label(self, name: str, is_box: bool) -> None:
        self.box_labels = [n for n in self.box_labels if n != name] + ([name] if is_box and name in self.order else []); self.save()

    def add_ref(self, frame: int) -> None:
        if frame not in self.refs:
            self.refs = sorted(self.refs + [int(frame)]); self.save()

    def add_closed(self, frame: int) -> None:
        if frame not in self.closed:
            self.closed = sorted(self.closed + [int(frame)]); self.save()

    def add_seed(self, name: str, frame: int, rect: Rect, auto_rects: list | None = None) -> int:
        """加一個手框的種子；auto_rects＝[(格, rect), ...] 是它往前後自動追出來的框，同一組。回傳組號。"""
        name = name.strip()
        if name not in self.order:
            self.order.append(name)
        g = max([s.group for s in self.seeds] + [0]) + 1
        self.seeds.append(Seed(name, int(frame), [int(v) for v in rect], False, g))
        for f, r in auto_rects or []:
            self.seeds.append(Seed(name, int(f), [int(v) for v in r], True, g))
        self.save()
        return g

    def remove_seed(self, index: int) -> None:
        """刪一個框：手框的會連同它自動追出來的整組一起刪；自動的只刪那一格。"""
        if not (0 <= index < len(self.seeds)):
            return
        sd = self.seeds[index]
        if not sd.auto and sd.group:
            self.seeds = [s for s in self.seeds if s.group != sd.group]
        else:
            self.seeds.pop(index)
        self.save()

    def manual(self) -> list:
        return [s for s in self.seeds if not s.auto]

    def remove_item(self, name: str) -> None:
        self.seeds = [s for s in self.seeds if s.name != name]
        self.order = [n for n in self.order if n != name]
        self.box_labels = [n for n in self.box_labels if n != name]; self.save()

    def counts(self) -> dict[str, int]:
        """每樣物品手框了幾次（自動追出來的不算，另見 auto_counts）。"""
        c: dict[str, int] = {}
        for s in self.seeds:
            if not s.auto:
                c[s.name] = c.get(s.name, 0) + 1
        return c

    def auto_counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for s in self.seeds:
            if s.auto:
                c[s.name] = c.get(s.name, 0) + 1
        return c


# ---------------------------------------------------------------- 框選後自動往前後追

TRACK_STEP = 3                  # 每幾格追一次（30 fps＝0.1 秒）
TRACK_KEEP_MIN = 0.55           # 跟「手框的那一格」比的最低分數
TRACK_KEEP_LAST = 0.75          # 或跟「上一個追到的位置」比的最低分數（物品慢慢轉角度、被手遮一點時靠這個接上）
TRACK_COLOR_MIN = 0.5           # 追到的框跟手框那一格的顏色（HSV 色相×飽和度直方圖相關）低於這個就當沒追到：
                                # 灰階樣板比對分不出「藍盒翻成牛皮色盒底」，名稱就錯了。現場 34 個手框實測（逐格人工判過 559 格）：
                                # 不檢查 467 對／92 錯；0.5 → 419／13；0.6 → 388／8；0.7 → 345／4。只限制移動距離幾乎沒用（451／85）


def track_seed(video_path: str, frame: int, rect: Rect, window_sec: float = 2.0, fps: float | None = None,
               step: int = TRACK_STEP) -> list[tuple[int, list[int]]]:
    """從手框的那一格往前、往後各追 window_sec 秒：每 step 格在上一個位置附近（框外擴 50%）做樣板比對，
    分數夠（比原框 ≥ 0.55 或比上一個位置 ≥ 0.75）就記一個框，連續 2 次不夠就停（物品離開／被遮住／換成別的樣子）。
    回傳 [(格, [x, y, w, h])]，不含手框那一格。影片只循序讀一次那段範圍（mp4 往回跳很慢）。"""
    from video_recorder import open_video_oriented
    cap, rotate, W, H, vfps = open_video_oriented(video_path)
    fps = float(fps or vfps or 30.0); n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    half = int(round(window_sec * fps)); a = max(0, frame - half); b = min(max(n_total - 1, frame), frame + half)
    from core_detector import roi_similarity
    grays: dict[int, np.ndarray] = {}
    smalls: dict[int, np.ndarray] = {}           # 彩色半尺寸（顏色檢查用；整張彩色存 40 格太吃記憶體）
    cap.set(cv2.CAP_PROP_POS_FRAMES, a); i = a
    while i <= b:
        ok, fr = cap.read()
        if not ok:
            break
        if (i - frame) % step == 0:
            if rotate is not None:
                fr = cv2.rotate(fr, rotate)
            grays[i] = _gray(fr)
            smalls[i] = cv2.resize(fr, (fr.shape[1] // 2, fr.shape[0] // 2), interpolation=cv2.INTER_AREA)
        i += 1
    cap.release()
    if frame not in grays:
        return []
    x, y, w, h = [int(v) for v in rect]
    tpl0 = grays[frame][max(0, y):y + h, max(0, x):x + w]
    if tpl0.size == 0 or tpl0.shape[0] < 8 or tpl0.shape[1] < 8:
        return []

    def color(f, r):
        rx, ry, rw, rh = r
        return smalls[f][max(0, ry // 2):(ry + rh) // 2, max(0, rx // 2):(rx + rw) // 2]
    col0 = color(frame, (x, y, w, h))
    out: list[tuple[int, list[int]]] = []
    for direction in (1, -1):
        last = (x, y, w, h); tpl_last = tpl0; misses = 0
        f = frame + direction * step
        while f in grays:
            g = grays[f]; lx, ly, lw, lh = last
            sx1, sy1 = max(0, int(lx - lw * 0.5)), max(0, int(ly - lh * 0.5))
            sx2, sy2 = min(g.shape[1], int(lx + lw * 1.5)), min(g.shape[0], int(ly + lh * 1.5))
            sub = g[sy1:sy2, sx1:sx2]
            s0, r0 = _track(tpl0, sub)
            s1, r1 = _track(tpl_last, sub)
            pick = None
            if s0 >= TRACK_KEEP_MIN and (s0 >= s1 or s1 < TRACK_KEEP_LAST):
                pick = r0
            elif s1 >= TRACK_KEEP_LAST:
                pick = r1
            if pick is not None:
                cand = [int(sx1 + pick[0]), int(sy1 + pick[1]), int(pick[2]), int(pick[3])]
                if roi_similarity(col0, color(f, cand)) < TRACK_COLOR_MIN:
                    pick = None                          # 形狀像、顏色不像：多半是翻面或換成別的東西
            if pick is None:
                misses += 1
                if misses >= 2:
                    break
            else:
                misses = 0
                nr = [int(sx1 + pick[0]), int(sy1 + pick[1]), int(pick[2]), int(pick[3])]
                out.append((f, nr)); last = tuple(nr)
                tpl_last = g[nr[1]:nr[1] + nr[3], nr[0]:nr[0] + nr[2]]
            f += direction * step
    out.sort(key=lambda t: t[0])
    return out


# ---------------------------------------------------------------- 追蹤（盒子＋手）

def trace_fingerprint(video_path: str) -> str:
    st = os.stat(video_path)
    return f"{os.path.basename(video_path)}|{st.st_size}|{int(st.st_mtime)}"


def build_trace(project, hand_model: str, yolo_weights: str, progress=None, force: bool = False) -> dict:
    """整支示範影片：盒子偵測每 DET_EVERY 格、手部每格（存 TRACE_STEP 的倍數格）。有快取且指紋相同就直接用。"""
    path = os.path.join(project.dir_path, TRACE_FILE)
    fp = trace_fingerprint(project.video_path)
    if not force and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if d.get("fingerprint") == fp and d.get("prompts") == [a.prompt for a in project.anchors]:
            return d
    from core_detector import HandTracker
    from object_detector import ObjectDetector
    from video_recorder import VideoFileSource
    det = ObjectDetector(yolo_weights, [a.prompt for a in project.anchors], conf=min(a.min_conf for a in project.anchors) if project.anchors else 0.2)
    hands = HandTracker(hand_model)
    src = VideoFileSource(project.video_path)
    rows = []; n = 0; total = src.frame_count
    try:
        while True:
            fr = src.read()
            if fr is None:
                break
            t = src.time_sec; row = {"f": n}
            if n % DET_EVERY == 0:
                row["dets"] = [[d.label, round(d.conf, 3)] + [round(float(v), 1) for v in d.xyxy] for d in det.detect(fr)]
            pts = hands.detect(fr, int(t * 1000))
            if n % TRACE_STEP == 0:
                row["hands"] = [[round(float(x), 1), round(float(y), 1)] for x, y in pts]
            if len(row) > 1:
                rows.append(row)
            n += 1
            if progress and n % 30 == 0:
                progress(n, total)
    finally:
        hands.close(); src.release()
    d = {"fingerprint": fp, "prompts": [a.prompt for a in project.anchors], "fps": project.fps, "frames": n, "rows": rows}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    if progress:
        progress(total, total)
    return d


@dataclass
class FrameStatus:
    region: Rect | None
    free: bool                  # 盒子新鮮、停好、手不在盒內
    placement: int              # 第幾次擺放（盒子出現或移動就 +1）
    judgeable: bool = False     # 盒子新鮮、停好（手在不在都算）：物品延伸與偵測用這個，手一直在盒內也要有資料


def replay(project, trace: dict) -> dict[int, FrameStatus]:
    """把 trace 用監控同一套規則重放，得到每個取樣格的盒內區域、能不能判、屬於第幾次擺放。"""
    from core_detector import STATE_FRESH_SEC, STATE_SETTLE_SEC, hands_in_inner, state_region
    from scene import Detection, SceneTracker, rect_overlap_ratio
    rows = {r["f"]: r for r in trace["rows"]}
    fps = float(trace.get("fps") or project.fps or 30.0); n_frames = int(trace["frames"])
    sc = SceneTracker(project.anchors, tuple(project.work_area) if project.work_area else None)
    out: dict[int, FrameStatus] = {}
    last_hands: list = []; region = None; moved_at = -10 ** 9; settle = int(round(STATE_SETTLE_SEC * fps)); placement = 0
    for f in range(n_frames):
        r = rows.get(f)
        if r and "dets" in r:
            sc.update_detections([Detection(d[0], d[1], tuple(d[2:6])) for d in r["dets"]], f / fps)
        if r and "hands" in r:
            last_hands = [tuple(h) for h in r["hands"]]
        new = state_region(project, sc.anchors_at(f / fps))
        if new is None:
            if region is not None:
                placement += 1
            region = None
        else:
            if region is None or min(rect_overlap_ratio(new, region), rect_overlap_ratio(region, new)) < 0.6:
                nested = region is not None and max(rect_overlap_ratio(new, region), rect_overlap_ratio(region, new)) >= 0.9
                if not nested:
                    moved_at = f; placement += 1
            region = new
        if f % TRACE_STEP == 0:
            if is_fixed(project):
                # 固定桌面範圍：每一格都能判（監控時也是不管盒子在不在、每 N 格就偵測）
                judgeable = True
            else:
                age = sc.age(project.state_anchor, f / fps) if project.state_anchor else 0.0
                judgeable = region is not None and age is not None and age <= STATE_FRESH_SEC and (f - moved_at) >= settle
            free = judgeable and not hands_in_inner(last_hands, region)
            out[f] = FrameStatus(tuple(int(v) for v in region) if region else None, free, placement, judgeable)
    return out


def region_for_frame(status: dict[int, FrameStatus], frame: int) -> Rect | None:
    """任一格（不一定是取樣格）的盒內區域：取最近的取樣格。"""
    f = frame - frame % TRACE_STEP
    for d in (0, -TRACE_STEP, TRACE_STEP, -2 * TRACE_STEP, 2 * TRACE_STEP):
        st = status.get(f + d)
        if st is not None and st.region is not None:
            return st.region
    return None


# ---------------------------------------------------------------- 延伸成資料集

def _crop(frame_bgr, rect: Rect):
    x, y, w, h = [int(round(v)) for v in rect]
    fh, fw = frame_bgr.shape[:2]
    return frame_bgr[max(0, y):min(fh, y + h), max(0, x):min(fw, x + w)]


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _iou(a: Rect, b: Rect) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _track(tpl, g, scales=(0.9, 1.0, 1.1)):
    best = (-1.0, None)
    for s in scales:
        t = cv2.resize(tpl, None, fx=s, fy=s) if s != 1.0 else tpl
        if t.shape[0] < 8 or t.shape[1] < 8 or t.shape[0] >= g.shape[0] or t.shape[1] >= g.shape[1]:
            continue
        res = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx > best[0]:
            best = (float(mx), (int(loc[0]), int(loc[1]), int(t.shape[1]), int(t.shape[0])))
    return best


def _labels_fixed(project, status: dict, seeds: ItemSeeds, reader, progress=None) -> tuple[dict, dict]:
    """固定桌面範圍的延伸：每個種子只在前後 FIXED_WINDOW_SEC 內用樣板比對（位置要跟框選時重疊）；種子那一格本身一定標；
    參考格＝離它最近的「空桌面」格（比對分數要比在空桌面上高 MATCH_OVER_EMPTY 才算）；空桌面格本身當背景圖（沒有標籤）寫進去，
    模型才學得到「桌上沒東西」。"""
    reg = tuple(project.state_fixed); fps = float(project.fps or 30.0); win = int(FIXED_WINDOW_SEC * fps)
    frames_sorted = sorted(status)
    ref_cache: dict[int, object] = {}

    def ref_gray(f):
        if not seeds.refs:
            return None
        r = min(seeds.refs, key=lambda x: abs(x - f))
        if r not in ref_cache:
            ref_cache[r] = _gray(_crop(reader.get_frame(r), reg))
        return ref_cache[r]

    direct, prop = split_seeds(seeds)
    tpls = []
    for sd in prop:
        x, y, w, h = sd.rect
        tpl = _gray(reader.get_frame(sd.frame))[max(0, y):y + h, max(0, x):x + w]
        if tpl.size == 0:
            continue
        rg = ref_gray(sd.frame)
        tpls.append((sd, tpl, _track(tpl, rg)[0] if rg is not None else -1.0, (x - reg[0], y - reg[1], w, h)))
    labels: dict[int, list] = {}; per_item: dict[str, int] = {}

    def add(f, name, r, score):
        found = labels.setdefault(f, [])
        dup = next((i for i, (n2, r2, s2) in enumerate(found) if n2 == name and _iou(r, r2) > 0.5), None)
        if dup is not None:
            if found[dup][2] >= score:
                return
            found.pop(dup)
        found.append((name, r, score))

    for sd, _, _, expected in tpls:            # 種子格本身：照框選的位置標（不用比對）
        add(sd.frame, sd.name, expected, 2.0)
    for sd in direct:                           # 框選當下就追好、使用者看過的框：直接當標籤，不再延伸（不然刪掉的錯框又會長回來）
        x, y, w, h = sd.rect
        r = _clip_rect((x - reg[0], y - reg[1], w, h), reg[2], reg[3])
        if r is not None:
            add(sd.frame, sd.name, r, 1.5)
    todo = [f for f in frames_sorted if any(abs(f - sd.frame) <= win for sd, _, _, _ in tpls)]
    for k, f in enumerate(todo):
        g = _gray(_crop(reader.get_frame(f), reg))
        for sd, tpl, ref_score, expected in tpls:
            if abs(f - sd.frame) > win or f == sd.frame:
                continue
            score, r = _track(tpl, g)
            if r is None or score < MATCH_MIN or score - ref_score < MATCH_OVER_EMPTY or _iou(r, expected) < MATCH_MIN_IOU:
                continue
            add(f, sd.name, r, score)
        if progress and k % 20 == 0:
            progress(k, len(todo))
    for r in seeds.refs:
        labels.setdefault(r, [])
    out = {}
    for f, found in labels.items():
        img = _crop(reader.get_frame(f), reg)
        if img.size == 0:
            continue
        out[f] = (img, [(n, r) for n, r, _ in found])
        for n, _, _ in found:
            per_item[n] = per_item.get(n, 0) + 1
    return out, per_item


def build_dataset(project, trace: dict, seeds: ItemSeeds, progress=None, val_fraction: float = 0.15) -> dict:
    """種子 → 同一次擺放內逐格延伸 → <專案>/items/dataset（YOLO 格式）。回傳統計。"""
    from video_recorder import FrameReader
    status = replay(project, trace)
    names = list(seeds.order)
    ds = dataset_dir(project)
    shutil.rmtree(ds, ignore_errors=True)
    for s in ("train", "val"):
        os.makedirs(os.path.join(ds, "images", s)); os.makedirs(os.path.join(ds, "labels", s))
    reader = FrameReader(project.video_path)
    if is_fixed(project):
        try:
            labels, per_item = _labels_fixed(project, status, seeds, reader, progress)
        finally:
            reader.release()
        return _write_dataset(project, ds, names, labels, per_item, {"placements": 1}, status, seeds, val_fraction)
    frames_sorted = sorted(status)
    # 每個種子：所屬擺放、空盒參考（同擺放中最早的參考格；沒有就該擺放第一個可判定的格）
    by_place: dict[int, list] = {}
    direct, prop = split_seeds(seeds)
    for sd in prop:
        f5 = sd.frame - sd.frame % TRACE_STEP
        st = status.get(f5)
        if st is None or st.region is None:
            continue
        by_place.setdefault(st.placement, []).append(sd)
    labels: dict[int, list] = {}; per_item: dict[str, int] = {}
    total_places = max(1, len(by_place)); done = 0
    for place, sds in by_place.items():
        # 延伸到手在盒內的格也要（監控時手在也會判）；樣板被手遮太多分數自然不夠就不會標
        frames = [f for f in frames_sorted if status[f].placement == place and status[f].judgeable]
        free_frames = [f for f in frames if status[f].free]
        if not frames:
            continue
        ref_frame = next((r for r in seeds.refs if status.get(r - r % TRACE_STEP) and status[r - r % TRACE_STEP].placement == place),
                         (free_frames or frames)[0])
        ref_reg = region_for_frame(status, ref_frame)
        ref_g = _gray(_crop(reader.get_frame(ref_frame), ref_reg)) if ref_reg else None
        tpls = []
        for sd in sds:
            reg = region_for_frame(status, sd.frame)
            if reg is None:
                continue
            fr = reader.get_frame(sd.frame)
            x, y, w, h = sd.rect
            tpl = _gray(fr)[max(0, y):y + h, max(0, x):x + w]
            if tpl.size == 0:
                continue
            ref_score = _track(tpl, ref_g)[0] if ref_g is not None else -1.0
            tpls.append((sd.name, tpl, ref_score, (x, y, w, h)))
        for f in frames:
            reg = status[f].region
            img = _crop(reader.get_frame(f), reg)
            if img.size == 0:
                continue
            g = _gray(img); found = []
            for name, tpl, ref_score, srect in tpls:
                score, r = _track(tpl, g)
                if r is None or score < MATCH_MIN or score - ref_score < MATCH_OVER_EMPTY:
                    continue
                expected = (srect[0] - reg[0], srect[1] - reg[1], srect[2], srect[3])    # 框選位置換算到這一格的盒內座標
                if _iou(r, expected) < MATCH_MIN_IOU:
                    continue
                # 同名、位置重疊的（同一樣東西框了兩次）只留分數高的
                dup = next((i for i, (n2, r2, s2) in enumerate(found) if n2 == name and _iou(r, r2) > 0.5), None)
                if dup is not None:
                    if found[dup][2] >= score:
                        continue
                    found.pop(dup)
                found.append((name, r, score))
            if found:
                labels[f] = (img, [(n, r) for n, r, _ in found])
                for name, _, _ in found:
                    per_item[name] = per_item.get(name, 0) + 1
        done += 1
        if progress:
            progress(done, total_places)
    # 框選當下就追好的框：用那一格的盒內區域換算後直接當標籤
    for sd in direct:
        reg = region_for_frame(status, sd.frame)
        if reg is None:
            continue
        x, y, w, h = sd.rect; r = _clip_rect((x - reg[0], y - reg[1], w, h), reg[2], reg[3])
        if r is None:
            continue
        if sd.frame in labels:
            img, found = labels[sd.frame]
            if any(n2 == sd.name and _iou(r, r2) > 0.5 for n2, r2 in found):
                continue
            found.append((sd.name, r))
        else:
            img = _crop(reader.get_frame(sd.frame), reg)
            if img.size == 0:
                continue
            labels[sd.frame] = (img, [(sd.name, r)])
        per_item[sd.name] = per_item.get(sd.name, 0) + 1
    reader.release()
    return _write_dataset(project, ds, names, labels, per_item, {"placements": len(by_place)}, status, seeds, val_fraction)


def _clip_rect(r, W, H, min_frac: float = 0.4):
    """框裁到 [0, W)×[0, H) 裡面；裁掉超過 60% 就不要（物品大半在範圍外）。"""
    x, y, w, h = r
    x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x2 - x1 < 8 or y2 - y1 < 8 or (x2 - x1) * (y2 - y1) < min_frac * w * h:
        return None
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))


def split_seeds(seeds: ItemSeeds) -> tuple[list, list]:
    """(直接當標籤的, 要延伸的)：框選當下有追過的那一組（含手框的那一個）直接用；舊資料（沒追過）照原本方式延伸。"""
    tracked = {s.group for s in seeds.seeds if s.auto and s.group}
    direct = [s for s in seeds.seeds if s.auto or (s.group and s.group in tracked)]
    prop = [s for s in seeds.seeds if not s.auto and not (s.group and s.group in tracked)]
    return direct, prop


def _write_dataset(project, ds: str, names: list[str], labels: dict, per_item: dict, extra: dict, status: dict, seeds: ItemSeeds,
                   val_fraction: float) -> dict:
    # 切訓練／驗證：隨機抽 val_fraction 當驗證（之前用「最後一次擺放」會把 1/3 的資料拿去驗證，使用者的 150 張只剩 99 張能訓）
    rng = np.random.RandomState(0); n_tr = n_val = 0
    for f, (img, found) in labels.items():
        is_val = rng.rand() < val_fraction
        split = "val" if is_val else "train"
        H, W = img.shape[:2]; nm = f"f{f:06d}"
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])   # cv2.imwrite 在 Windows 對非 ASCII 路徑會靜默失敗
        buf.tofile(os.path.join(ds, "images", split, nm + ".jpg"))
        if found:                                                           # 沒有標籤的圖＝背景（空桌面），不寫 txt
            with open(os.path.join(ds, "labels", split, nm + ".txt"), "w") as fh:
                for name, (x, y, w, h) in found:
                    fh.write(f"{names.index(name)} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}\n")
        if is_val:
            n_val += 1
        else:
            n_tr += 1
    with open(os.path.join(ds, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write(f"path: {ds}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    wa = suggest_work_area(project, status, seeds.refs)
    if wa is not None:
        with open(os.path.join(project.dir_path, WORK_AREA_FILE), "w", encoding="utf-8") as fh:
            json.dump(wa, fh)
    return {"train": n_tr, "val": n_val, "per_item": per_item, "names": names, "work_area": wa, **extra}


WORK_AREA_FILE = os.path.join(ITEMS_DIR, "work_area.json")


def suggest_work_area(project, status: dict, refs: list[int], margin: float = 0.2) -> list[int] | None:
    """用空盒參考格的盒子位置推工作範圍：取聯集再往外擴 margin 倍、夾在畫面內。畫面裡別處的紙箱（架上的）就不會被當成盒子。
    margin 0.4 時桌面測試右上角架上的紙箱（中心 x 1537）還在範圍內、第 2 輪結不了；0.2 剛好排除，盒子每輪位置差異仍蓋得到。
    固定桌面範圍時就是那個範圍本身（盒子的中心要落在桌面上才算；現場畫面下方堆的鞋盒、上方架上的紙箱都在範圍外）。"""
    if is_fixed(project):
        return [int(v) for v in project.state_fixed]
    rects = [region_for_frame(status, f) for f in refs]
    rects = [r for r in rects if r is not None]
    if not rects:
        return None
    x1 = min(r[0] for r in rects); y1 = min(r[1] for r in rects)
    x2 = max(r[0] + r[2] for r in rects); y2 = max(r[1] + r[3] for r in rects)
    w, h = x2 - x1, y2 - y1
    W, H = project.resolution
    X1 = max(0, int(x1 - w * margin)); Y1 = max(0, int(y1 - h * margin))
    X2 = min(W, int(x2 + w * margin)); Y2 = min(H, int(y2 + h * margin))
    return [X1, Y1, X2 - X1, Y2 - Y1]


class OpenClosedClassifier:
    """盒子是開著還是蓋著：CLIP 最近鄰，開著＝空盒參考格、蓋著＝使用者標的蓋著格（都用當時的盒子區域裁圖）。
    沒標過蓋著的格就不啟用（is_open 回 None）。"""

    def __init__(self, project, seeds: "ItemSeeds", embedder=None):
        self.vecs: list = []; self.labels: list[bool] = []
        if not seeds.closed or not seeds.refs:
            return
        trace_path = os.path.join(project.dir_path, TRACE_FILE)
        if not os.path.isfile(trace_path):
            return
        from video_recorder import FrameReader
        with open(trace_path, "r", encoding="utf-8") as f:
            trace = json.load(f)
        status = replay(project, trace)
        if embedder is None:
            from state_recognizer import Embedder
            embedder = Embedder()
        self.embedder = embedder
        reader = FrameReader(project.video_path)
        # 「開著」的範例要同時有空盒（參考格）和放了東西的盒子（框選物品的那些格），不然裝滿黑鞋的盒子會被判成蓋著
        # （桌面測試第 2 輪：鞋子一放進去 1 秒就被當成蓋上）
        open_frames = sorted(set(seeds.refs) | {sd.frame for sd in seeds.manual()})
        try:
            for frames, is_open in ((open_frames, True), (seeds.closed, False)):
                for fr in frames:
                    reg = region_for_frame(status, fr)
                    if reg is None:
                        continue
                    crop = _crop(reader.get_frame(fr), reg)
                    if crop.size:
                        self.vecs.append(self.embedder.embed(crop)); self.labels.append(is_open)
        finally:
            reader.release()
        if not any(self.labels) or all(self.labels):
            self.vecs, self.labels = [], []

    @property
    def enabled(self) -> bool:
        return bool(self.vecs)

    def is_open(self, crop_bgr) -> bool | None:
        if not self.vecs or crop_bgr is None or crop_bgr.size == 0:
            return None
        v = self.embedder.embed(crop_bgr)
        d = np.linalg.norm(np.array(self.vecs) - v, axis=1)
        return bool(self.labels[int(d.argmin())])


def train_command(project, epochs: int = 30, imgsz: int | None = None) -> list[str]:
    """給介面用子程序跑的指令（在 UI 行程裡訓練會把記憶體與 Qt 攪在一起）。imgsz 不給就依模式：盒內 640、固定桌面 960。"""
    import sys
    root = os.path.dirname(os.path.abspath(__file__))
    if imgsz is None:
        imgsz = FIXED_IMGSZ if is_fixed(project) else 640
    return [sys.executable, os.path.join(root, "tools", "train_items.py"), os.path.join(dataset_dir(project), "data.yaml"),
            train_dir(project), "--epochs", str(epochs), "--imgsz", str(imgsz), "--copy-to", os.path.join(project.dir_path, MODEL_FILE)]


# ---------------------------------------------------------------- 監控用偵測器

@dataclass
class ItemHit:
    name: str
    conf: float
    rect: Rect                  # 盒內裁圖座標


class ItemDetector:
    """訓練好的小模型：盒內裁圖 → 哪些物品在哪裡。"""

    def __init__(self, model_path: str, conf: float = 0.3, imgsz: int | None = None, threads: int = 2):
        # conf 0.3：清單規則要連續 0.5 秒看到才算，偶發的誤框撐不過；0.5 時使用者自己訓的模型鞋子、標籤常漏（實測）
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(max(1, threads))
        pt = YOLO(model_path)
        if imgsz is None:
            # 用訓練時的尺寸（固定桌面範圍是 960）；拿不到就 640
            try:
                imgsz = int((pt.ckpt or {}).get("train_args", {}).get("imgsz") or 640)
            except Exception:
                imgsz = 640
        self.conf = conf; self.imgsz = imgsz
        self.backend = "pytorch"
        self.model = pt
        ov = openvino_model(model_path, imgsz)
        if ov is not None:
            try:
                m = YOLO(ov, task="detect")
                m.predict(np.zeros((imgsz, imgsz, 3), np.uint8), imgsz=imgsz, conf=conf, verbose=False)
                self.model = m; self.backend = "openvino"
            except Exception:
                pass
        self.names = [self.model.names[i] for i in sorted(self.model.names)] if isinstance(self.model.names, dict) else list(self.model.names)

    def detect(self, crop_bgr) -> list[ItemHit]:
        if crop_bgr is None or crop_bgr.size == 0:
            return []
        r = self.model.predict(crop_bgr, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            out.append(ItemHit(self.model.names[int(b.cls[0])], float(b.conf[0]), (int(x1), int(y1), int(x2 - x1), int(y2 - y1))))
        return out


def model_path(project) -> str:
    return os.path.join(project.dir_path, MODEL_FILE)


def openvino_model(pt_path: str, imgsz: int) -> str | None:
    """把訓練好的 best.pt 轉成 OpenVINO（Intel CPU 專用推論格式），回傳轉好的資料夾；沒裝 openvino 或轉失敗回 None（照用 PyTorch）。
    現場專案 960 實測：PyTorch 446 ms → OpenVINO 61 ms，40 張抽樣認出的東西完全相同。轉檔約 11 秒，結果快取在
    cache/ov_<模型路徑＋修改時間的雜湊>/（純 ASCII 路徑；模型重訓後修改時間變了就自動重轉）。"""
    import hashlib
    try:
        import openvino  # noqa: F401
    except Exception:
        return None
    try:
        st = os.stat(pt_path)
        key = hashlib.md5(f"{os.path.abspath(pt_path)}|{int(st.st_mtime)}|{st.st_size}|{imgsz}".encode("utf-8")).hexdigest()[:12]
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "ov_" + key)
        done = os.path.join(root, "best_openvino_model")
        if os.path.isfile(os.path.join(done, "best.xml")):
            return done
        os.makedirs(root, exist_ok=True)
        shutil.copyfile(pt_path, os.path.join(root, "best.pt"))
        from ultralytics import YOLO
        out = YOLO(os.path.join(root, "best.pt")).export(format="openvino", imgsz=imgsz, half=False, verbose=False)
        return out if os.path.isdir(out) else None
    except Exception:
        return None
