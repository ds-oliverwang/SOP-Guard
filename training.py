"""「示範影片＋分段標記 → 手勢模型 → 專案步驟」的服務層，給介面與命令列共用。

流程：
  1. 對專案的示範影片抽手部關鍵點（逐格）與 gesture_anchors 的物件框（每 15 格），快取到 <專案>/gestures/_cache/demo.npz
  2. 依 segments.json 切片段，沒標的空隙當「其他」
  3. 依片段分組交叉驗證、用全部資料訓練、存 gesture_model.pkl
  4. 把每個動作（「其他」除外）寫成一個 gesture 步驟（保留既有的非 gesture 步驟），cycle_end 若未設則設成「手離開畫面 3 秒」
"""
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from gesture_classifier import GestureClassifier, N_ANCHORS
from gesture_data import Clip, anchor_rects_per_frame, build_dataset, clips_from_segments, cross_validate_by_clip, format_report
from project_store import Project, load_project, project_to_dict

SEGMENTS_FILE = "gestures/segments.json"
CACHE_FILE = "gestures/_cache/demo.npz"
OTHER_LABEL = "其他"
END_LABELS = ("結束", "完成", "结束")   # 分段標成這些名字 → 當成「一輪結束」的手勢，不當步驟


@dataclass
class TrainResult:
    accuracy: float
    recall: dict[str, float]
    report: str
    labels: list[str]
    windows_per_class: dict[str, int] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    model_path: str = ""


def segments_path(project: Project) -> str:
    return os.path.join(project.dir_path, SEGMENTS_FILE)


def load_segments(project: Project) -> list[dict]:
    p = segments_path(project)
    if not os.path.isfile(p):
        return []
    with open(p, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return [s for s in data.get("segments", []) if isinstance(s, dict)]


def save_segments(project: Project, segments: list[dict]) -> None:
    p = segments_path(project)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"video_file": project.video_file, "segments": segments}, f, ensure_ascii=False, indent=2)


def video_fingerprint(video_path: str) -> str:
    """快取要認得「是哪一支影片」：檔名＋大小＋修改時間＋手部偵測門檻。匯入舊檔案時修改時間可能比快取還舊，不能只比新舊。"""
    from core_detector import HAND_DET_CONF
    st = os.stat(video_path)
    return f"{os.path.basename(video_path)}|{st.st_size}|{int(st.st_mtime)}|conf{HAND_DET_CONF}"


def extract_demo_features(project: Project, hand_model: str, yolo_weights: str | None, progress=None, force: bool = False) -> dict:
    """抽示範影片的特徵並快取。progress(done, total) 可選。回傳 {"hands","det","labels","fps"}。"""
    cache = os.path.join(project.dir_path, CACHE_FILE)
    video = project.video_path
    if not os.path.isfile(video):
        raise FileNotFoundError(f"找不到示範影片：{video}")
    anchor_prompts = [next(a.prompt for a in project.anchors if a.name == n) for n in project.gesture_anchors][:N_ANCHORS]
    fp = video_fingerprint(video)
    if not force and os.path.isfile(cache):
        d = np.load(cache, allow_pickle=False)
        labels = [str(s) for s in d["labels"]]
        cached_fp = str(d["fingerprint"]) if "fingerprint" in d.files else ""
        if labels == anchor_prompts and cached_fp == fp:
            return {"hands": d["hands"], "det": d["det"], "labels": labels, "fps": float(d["fps"])}
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    from video_recorder import VideoFileSource
    yolo = None
    if anchor_prompts:
        if not yolo_weights:
            raise ValueError("專案有 gesture_anchors，需要 YOLO-World 權重")
        from object_detector import ObjectDetector
        yolo = ObjectDetector(yolo_weights, anchor_prompts)
    from core_detector import HAND_DET_CONF
    opts = vision.HandLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=hand_model),
                                        running_mode=vision.RunningMode.VIDEO, num_hands=2,
                                        min_hand_detection_confidence=HAND_DET_CONF, min_tracking_confidence=HAND_DET_CONF)
    det = vision.HandLandmarker.create_from_options(opts)
    src = VideoFileSource(video)
    W, H = src.resolution
    total = src.frame_count
    hands = []; dets = []; last_ts = -1; i = -1
    scale = 640 / max(W, H); dw, dh = int(W * scale), int(H * scale)
    try:
        while True:
            frame = src.read()
            if frame is None:
                break
            i += 1
            rgb = cv2.cvtColor(cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            ts = int(i / src.fps * 1000); ts = ts if ts > last_ts else last_ts + 1; last_ts = ts
            r = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            h = np.full((2, 21, 2), np.nan, np.float32)
            for k, hand in enumerate(r.hand_landmarks[:2]):
                h[k] = [(lm.x * W, lm.y * H) for lm in hand]
            hands.append(h)
            if yolo is not None and i % 15 == 0:
                for d in yolo.detect(frame):
                    dets.append([i, anchor_prompts.index(d.label), d.conf, *d.xyxy])
            if progress and i % 30 == 0:
                progress(i + 1, total)
    finally:
        src.release(); det.close()
    data = {"hands": np.array(hands, np.float32).reshape(-1, 2, 21, 2), "det": np.array(dets, np.float32).reshape(-1, 7),
            "labels": anchor_prompts, "fps": src.fps}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, hands=data["hands"], det=data["det"], labels=np.array(anchor_prompts), fps=np.float32(src.fps),
                        fingerprint=np.array(fp))
    if progress:
        progress(total, total)
    return data


def build_clips(project: Project, data: dict, segments: list[dict]) -> list[Clip]:
    n = len(data["hands"])
    rects = [anchor_rects_per_frame(data["det"], data["labels"], p, n) for p in data["labels"]]
    return clips_from_segments(data["hands"], rects, segments, other_label=OTHER_LABEL,
                               frame_size=(project.resolution[0], project.resolution[1]))


def propose_segments(data: dict, labeled: list[dict], min_len_frames: int = 30, prob_min: float = 0.3,
                     smooth: int = 9, stride: int = 5, frame_size=None) -> list[dict]:
    """用已標的分段訓一個草稿模型，掃影片沒標的部分，提出「建議分段」給人確認。

    回傳 [{"label","start","end","prob"}]，不含與已標分段重疊的範圍。沒標或只有一類時回傳空 list。
    做法：已標分段 + 空隙當「其他」→ RandomForest（100 棵）→ 每 stride 格算一次視窗機率 → 中位數平滑 →
    同一標籤連續 ≥ min_len_frames 且平均機率 ≥ prob_min 的區間當建議。"""
    from gesture_classifier import WINDOW, hand_frame_features, window_features, window_has_enough_hand
    hands = data["hands"]; n = len(hands)
    rects = [anchor_rects_per_frame(data["det"], data["labels"], p, n) for p in data["labels"]]
    # 「其他」只能取已標範圍內（含前後 2 秒）的空隙；再往後沒標的部分可能就是下一輪的動作，不能當「其他」學
    span_a = max(0, min(int(s["start"]) for s in labeled) - 60)
    span_b = min(n, max(int(s["end"]) for s in labeled) + 60)
    shifted = [{"label": s["label"], "start": int(s["start"]) - span_a, "end": int(s["end"]) - span_a} for s in labeled]
    clips = clips_from_segments(hands[span_a:span_b], [rl[span_a:span_b] for rl in rects], shifted, other_label=OTHER_LABEL,
                                frame_size=frame_size)
    if len({c.label for c in clips}) < 2:
        return []
    X, y, _g = build_dataset(clips)
    if len(y) == 0:
        return []
    clf = GestureClassifier(); clf.fit(X, y, n_estimators=100)
    classes = clf.classes
    feats = np.stack([hand_frame_features(hands[i], [rl[i] for rl in rects], frame_size) for i in range(n)])
    # 每個視窗中心一格的機率
    centers, probs = [], []
    for end in range(WINDOW, n + 1, stride):
        seg = feats[end - WINDOW:end]
        c = end - WINDOW // 2
        centers.append(c)
        if window_has_enough_hand(seg):
            probs.append(clf.model.predict_proba(window_features(seg).reshape(1, -1))[0])
        else:
            probs.append(np.zeros(len(classes)))
    if not centers:
        return []
    P = np.array(probs)
    if smooth > 1 and len(P) >= smooth:
        k = smooth // 2
        P = np.array([np.median(P[max(0, i - k):i + k + 1], axis=0) for i in range(len(P))])
    # 動作順序 = 第一輪標的順序（依起點排序、去重）；每一輪照這個順序循環
    order: list[str] = []
    for s in sorted(labeled, key=lambda s: int(s["start"])):
        if s["label"] not in order and s["label"] in classes and s["label"] != OTHER_LABEL:
            order.append(s["label"])
    if not order:
        return []
    example_len = {lab: float(np.median([int(s["end"]) - int(s["start"]) for s in labeled if s["label"] == lab])) for lab in order}
    path = _decode_cyclic_sequence(P, classes, order)
    # 連續同一動作狀態的區間 → 建議
    out = []
    i = 0
    while i < len(centers):
        j = i
        while j + 1 < len(centers) and path[j + 1] == path[i]:
            j += 1
        lab = path[i]
        if lab is not None:
            start = max(0, centers[i] - WINDOW // 2); end = min(n, centers[j] + WINDOW // 2)
            ci = classes.index(lab)
            mean_p = float(P[i:j + 1, ci].mean())
            long_enough = end - start >= max(min_len_frames, 0.4 * example_len.get(lab, min_len_frames))
            if long_enough and mean_p >= prob_min and overlapping_segment(labeled, start, end) is None:
                out.append({"label": lab, "start": int(start), "end": int(end), "prob": round(mean_p, 2)})
        i = j + 1
    return out


def _decode_cyclic_sequence(P: np.ndarray, classes: list[str], order: list[str],
                            p_stay: float = 0.95, floor: float = 0.02) -> list:
    """Viterbi：狀態為「其他₀ → 動作₁ → 其他₁ → 動作₂ → … → 動作ₖ → 其他ₖ → 回到動作₁」的循環，
    只允許停留或前進一步。回傳每個視窗的動作名稱（其他狀態為 None）。"""
    other_i = classes.index(OTHER_LABEL) if OTHER_LABEL in classes else None
    states: list = []          # (label or None)
    for lab in order:
        states.append(None); states.append(lab)
    states.append(None)
    S = len(states); T = len(P)
    # 發射機率：其他狀態用「其他」類的機率（沒有「其他」類時用 1 - max），動作狀態用該類機率
    def emit(t, s):
        lab = states[s]
        if lab is None:
            v = P[t, other_i] if other_i is not None else 1.0 - P[t].max()
        else:
            v = P[t, classes.index(lab)]
        return np.log(max(float(v), floor))
    log_stay, log_adv = np.log(p_stay), np.log(1 - p_stay)
    V = np.full((T, S), -np.inf); back = np.zeros((T, S), dtype=int)
    V[0, 0] = emit(0, 0)
    for t in range(1, T):
        for s in range(S):
            prev_same = V[t - 1, s] + log_stay
            prev_adv_state = (s - 1) if s > 0 else (S - 1)          # 從最後一個「其他」繞回第一個動作
            prev_adv = V[t - 1, prev_adv_state] + log_adv
            if s == 0:                                                # 其他₀ 只能從最後的其他ₖ 繞回（等同同一狀態），不另外前進
                prev_adv = V[t - 1, S - 1] + log_adv
            if prev_same >= prev_adv:
                V[t, s] = prev_same + emit(t, s); back[t, s] = s
            else:
                V[t, s] = prev_adv + emit(t, s); back[t, s] = prev_adv_state if s > 0 else S - 1
    s = int(V[T - 1].argmax()); path = [None] * T
    for t in range(T - 1, -1, -1):
        path[t] = states[s]; s = int(back[t, s])
    return path


def train_from_clips(clips: list[Clip], min_clips: int = 3, n_folds: int = 5) -> tuple[GestureClassifier | None, TrainResult]:
    per_label = Counter(c.label for c in clips)
    skipped = [l for l, n in per_label.items() if n < min_clips and l != OTHER_LABEL]
    clips = [c for c in clips if c.label not in skipped]
    labels = sorted({c.label for c in clips})
    if len(labels) < 2:
        detail = "、".join(f"「{l}」{per_label[l]} 段（還差 {max(0, min_clips - per_label[l])} 段）"
                          for l in per_label if l != OTHER_LABEL) or "（沒有任何動作分段）"
        msg = (f"每個動作至少要標 {min_clips} 段才會納入訓練，目前：{detail}。"
               f"請回時間軸多標幾段（影片太短就重錄一段 60 秒左右、每個動作做 3 次以上）。")
        return None, TrainResult(0.0, {}, msg, labels, {}, skipped)
    X, y, groups = build_dataset(clips)
    if len(y) == 0:
        return None, TrainResult(0.0, {}, "分段裡偵測不到手，無法訓練", labels, {}, skipped)
    r = cross_validate_by_clip(X, y, groups, n_folds=min(n_folds, max(2, min(per_label.values()))))
    clf = GestureClassifier(); clf.fit(X, y)
    return clf, TrainResult(r["accuracy"], r["recall"], format_report(r), labels, dict(r["windows_per_class"]), skipped)


def apply_gesture_steps(project: Project, labels: list[str], model_file: str, min_prob: float = 0.35,
                        dwell_sec: float = 0.5, hands_absent_hold: float = 3.0) -> None:
    """把動作寫成 gesture 步驟並存 config。手勢模式與物品／狀態模式二選一（使用者裁示）：舊的 item／box_state 步驟一併清掉，
    只保留 hand_dwell／object_in_zone 這類手動設定的步驟。
    一輪結束：有標「結束」手勢就用它（gesture 型，維持 0.5 秒）；否則保留既有設定；都沒有才用「手離開畫面」。"""
    from project_store import Step
    keep = [s for s in project.steps if s.type not in ("gesture", "item", "box_state")]
    steps = list(keep)
    end_label = next((l for l in labels if l in END_LABELS), None)
    seen: set[str] = set()
    for lab in labels:
        if lab == OTHER_LABEL or lab == end_label or lab in seen:
            continue
        seen.add(lab)
        steps.append(Step(step_index=0, step_name=lab, dwell_time_sec=dwell_sec, timeout_sec=60.0, type="gesture",
                          label=lab, min_prob=min_prob, required=True))
    for i, s in enumerate(steps):
        s.step_index = i
    project.steps = steps
    project.gesture_model = model_file
    project.schema_version = 2
    if end_label is not None:
        project.cycle_end = Step(step_index=-1, step_name="cycle_end", dwell_time_sec=0.5, timeout_sec=1.0, type="gesture",
                                 label=end_label, min_prob=min_prob)
    elif project.cycle_end is None or project.cycle_end.type == "gesture":
        project.cycle_end = Step(step_index=-1, step_name="cycle_end", dwell_time_sec=hands_absent_hold, timeout_sec=1.0, type="hands_absent")
    if project.cycle_min_sec <= 0:
        project.cycle_min_sec = 5.0
    cfg_path = os.path.join(project.dir_path, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(project_to_dict(project), f, ensure_ascii=False, indent=2)


def segment_counts(segments: list[dict], min_frames: int = 30) -> dict[str, int]:
    """每個動作「有效」幾段：與前面分段重疊的部分會被砍掉，砍完不足 min_frames 格的不算（和訓練時的切法一致）。"""
    segs = sorted(({"label": s["label"], "start": int(s["start"]), "end": int(s["end"])} for s in segments), key=lambda s: s["start"])
    counts: Counter = Counter()
    cursor = 0
    for s in segs:
        start = max(s["start"], cursor)
        if s["end"] - start >= min_frames:
            counts[s["label"]] += 1
        cursor = max(cursor, s["end"])
    return dict(counts)


def overlapping_segment(segments: list[dict], start: int, end: int) -> dict | None:
    """新分段 [start, end) 若與既有分段重疊，回傳那個既有分段。"""
    for s in segments:
        if start < int(s["end"]) and int(s["start"]) < end:
            return s
    return None


def propose_for_project(project_dir: str, hand_model: str, yolo_weights: str | None, progress=None) -> list[dict]:
    """介面用：抽（或讀快取）示範影片特徵，依目前已標的分段提出建議分段。"""
    project = load_project(project_dir)
    labeled = load_segments(project)
    if not labeled:
        return []
    data = extract_demo_features(project, hand_model, yolo_weights,
                                 progress=(lambda d, t: progress("抽特徵", d, t)) if progress else None)
    if progress:
        progress("找其他輪", 0, 1)
    out = propose_segments(data, labeled, frame_size=(project.resolution[0], project.resolution[1]))
    if progress:
        progress("完成", 1, 1)
    return out


STATE_END_LABELS = ("結束", "完成", "蓋上", "结束")


def ensure_state_anchor(project: Project, prompt: str = "cardboard box") -> bool:
    """盒內狀態需要「盒子」錨點；專案沒有就自動加一個（名稱 box），並把 state_anchor 指到它。回傳是否有改動。"""
    from scene import AnchorSpec
    changed = False
    if not any(a.name == "box" for a in project.anchors):
        # max_age 3 秒：盒子被手遮住幾秒狀態會維持，盒子真的拿走就不再對著空桌面判；max_count 2：盒蓋、盒身都留著讓黏住規則挑
        project.anchors.append(AnchorSpec(name="box", prompt=prompt, min_conf=0.2, pick="largest", max_count=2,
                                          min_area=20000.0, max_age_sec=3.0))
        changed = True
    if project.state_anchor is None:
        # 固定區域（state_fixed）時裁圖不用錨點，但物件版一輪結束仍靠「盒子被拿走」，錨點照設
        project.state_anchor = "box"; changed = True
    if changed:
        with open(os.path.join(project.dir_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(project_to_dict(project), f, ensure_ascii=False, indent=2)
    return changed


def sync_item_region(project: Project) -> bool:
    """物件版：把種子檔的「桌面範圍」(seeds.region) 套到 project.state_fixed（None＝跟著盒子裁）。建立訓練資料與產生步驟前都要套，
    延伸資料集（replay）跟監控用的是同一個欄位。回傳是否有改動（有就已寫回 config.json）。"""
    from item_recognizer import ItemSeeds
    seeds = ItemSeeds(project.dir_path)
    new = list(seeds.region) if seeds.region else None
    if new == (None if project.state_fixed is None else list(project.state_fixed)):
        return False
    project.state_fixed = new
    with open(os.path.join(project.dir_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(project_to_dict(project), f, ensure_ascii=False, indent=2)
    return True


def apply_state_steps(project: Project, order: list[str], dwell_sec: float = 0.5) -> list[str]:
    """把狀態順序寫成 box_state 步驟：第一個狀態是「起始」（不當步驟），名為 結束／完成／蓋上 的狀態當一輪結束，
    其餘依序各一個必做步驟。沒有結束狀態時，一輪結束＝盒子被拿走（錨點消失 2 秒）——內盒影片實測蓋上後 1～1.5 秒
    就被推走，「回到未開」常來不及認；盒子不見比較穩。保留既有的非 box_state 步驟。回傳步驟名稱。"""
    from project_store import Step
    order = [s for s in order if s]
    if not order:
        return []
    if project.state_anchor is None and project.state_fixed is None:
        ensure_state_anchor(project)     # box_state 步驟一定要有盒子錨點或固定區域
    initial = order[0]
    end_label = next((s for s in order if s in STATE_END_LABELS), None)
    steps = [s for s in project.steps if s.type != "box_state"]
    names = []
    for st in order:
        if st == initial or st == end_label:
            continue
        steps.append(Step(step_index=0, step_name=st, dwell_time_sec=dwell_sec, timeout_sec=120.0, type="box_state",
                          label=st, required=True))
        names.append(st)
    for i, s in enumerate(steps):
        s.step_index = i
    project.steps = steps
    project.schema_version = 2
    if end_label:
        project.cycle_end = Step(step_index=-1, step_name="cycle_end", dwell_time_sec=2.0, timeout_sec=1.0, type="box_state", label=end_label)
    else:
        project.cycle_end = Step(step_index=-1, step_name="cycle_end", dwell_time_sec=0.5, timeout_sec=1.0, type="object_absent",
                                 object=project.state_anchor or "box")
    if project.cycle_min_sec <= 0:
        project.cycle_min_sec = 5.0
    if project.state_every_n_frames > 5:
        project.state_every_n_frames = 5     # 舊專案預設 10 格一次，放完東西的空檔常不到 1 秒會來不及投票
    with open(os.path.join(project.dir_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(project_to_dict(project), f, ensure_ascii=False, indent=2)
    return names


def apply_item_steps(project: Project, names: list[str], dwell_sec: float = 0.5) -> list[str]:
    """物件版：每一樣物品一個必做步驟（type item，監控時在盒內看到過就算），一輪結束＝盒子被拿走 2 秒。保留非 item 步驟。"""
    from project_store import Step
    names = [n for n in names if n]
    from item_recognizer import ItemSeeds, WORK_AREA_FILE
    seeds = ItemSeeds(project.dir_path)
    sync_item_region(project)
    if project.state_anchor is None:
        ensure_state_anchor(project)
    if project.state_fixed is not None:
        # 看整張桌面（固定範圍）：物品偵測不跟著盒子；盒子錨點只管「推走＝一輪結束」，沿用時間縮到 1 秒
        # （現場影片兩輪只隔 2.7 秒，沿用 3 秒會把下一雙的翻鞋舌檢查算到上一輪）、下一輪 0.2 秒就能開始
        project.cycle_arm_clear_sec = 0.2
        for a in project.anchors:
            if a.name == project.state_anchor:
                a.max_age_sec = min(a.max_age_sec, 1.0)
    # 工作範圍：建立訓練資料時從空盒參考格推出來（架上別的紙箱不會被當成盒子）；每次產生步驟都重套（相機移了重跑第 2 步就會更新）
    wa_path = os.path.join(project.dir_path, WORK_AREA_FILE)
    if os.path.isfile(wa_path):
        with open(wa_path, "r", encoding="utf-8") as f:
            wa = json.load(f)
        if isinstance(wa, list) and len(wa) == 4:
            project.work_area = [int(v) for v in wa]
    # 物件版取代舊流程的全部步驟：舊的手勢步驟留著永遠不會被看到，每一輪都會報「漏做：放卡片…」（使用者的桌面測試專案實際踩到）
    # 「手停在物件上」的步驟（seeds.hand_steps）：同名的物品不再產生 item 步驟（例如檢查鞋舌改用手停在鞋上判）；
    # 物件錨點依提示自動補上（名稱＝提示；max_count 2 兩隻鞋都要、沿用 2.5 秒撐過手握住時掉框）
    from scene import AnchorSpec
    hand_names = {h["name"] for h in seeds.hand_steps}
    steps = []
    for n in names:
        if n in hand_names:
            continue
        steps.append(Step(step_index=0, step_name=n, dwell_time_sec=dwell_sec, timeout_sec=120.0, type="item", label=n, required=True))
    for h in seeds.hand_steps:
        if not any(a.name == h["prompt"] for a in project.anchors):
            # min_conf 0.15：通用提示對黑鞋信心偏低（現場影片第 2 輪檢查時 shoe 只有 0.10～0.23，0.25 整段沒錨點、整輪報漏做）；
            # 有工作範圍、面積下限、而且手還要落在框內，低門檻不會亂算
            project.anchors.append(AnchorSpec(name=h["prompt"], prompt=h["prompt"], min_conf=0.15, pick="largest", max_count=2,
                                              min_area=8000.0, max_age_sec=2.5))
        steps.append(Step(step_index=0, step_name=h["name"], dwell_time_sec=float(h["dwell"]), timeout_sec=120.0, type="hand_on_object",
                          object=h["prompt"], required=True, continuous=True))
        names = [n for n in names if n != h["name"]] + [h["name"]]
    project.gesture_model = None; project.gesture_anchors = []
    for i, s in enumerate(steps):
        s.step_index = i
    project.steps = steps
    project.schema_version = 2
    # 盒子錨點本身就會沿用 3 秒才算消失，結束條件只要再 0.5 秒去抖；之前設 2 秒，操作者 26.5 秒就把下一個盒子放上來
    # （拿走後 4 秒內），結束倒數被新盒子打斷，整輪永遠不結算（介面實跑抓到）
    project.cycle_end = Step(step_index=-1, step_name="cycle_end", dwell_time_sec=0.5, timeout_sec=1.0, type="object_absent",
                             object=project.state_anchor or "box")
    if project.cycle_min_sec <= 0:
        project.cycle_min_sec = 5.0
    if project.state_every_n_frames > 5:
        project.state_every_n_frames = 5
    with open(os.path.join(project.dir_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(project_to_dict(project), f, ensure_ascii=False, indent=2)
    return names


def train_project(project_dir: str, hand_model: str, yolo_weights: str | None, progress=None,
                  model_file: str = "gesture_model.pkl", write_steps: bool = True, min_clips: int = 3) -> TrainResult:
    """一鍵：抽特徵 → 切片段 → 訓練 → 存模型 → 寫步驟。progress(stage: str, done: int, total: int) 可選。"""
    project = load_project(project_dir)
    segments = load_segments(project)
    if not segments:
        return TrainResult(0.0, {}, "還沒有任何分段標記", [], {}, [])
    data = extract_demo_features(project, hand_model, yolo_weights,
                                 progress=(lambda d, t: progress("抽特徵", d, t)) if progress else None)
    clips = build_clips(project, data, segments)
    if progress:
        progress("訓練", 0, 1)
    clf, result = train_from_clips(clips, min_clips=min_clips)
    if clf is None:
        return result
    out = os.path.join(project.dir_path, model_file)
    clf.save(out)
    result.model_path = out
    if write_steps:
        # 步驟順序照操作者示範時第一次出現的順序
        order: list[str] = []
        for s in sorted(segments, key=lambda s: int(s["start"])):
            if s["label"] in result.labels and s["label"] not in order:
                order.append(s["label"])
        apply_gesture_steps(project, order, model_file)
    if progress:
        progress("完成", 1, 1)
    return result
