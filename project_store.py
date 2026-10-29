"""專案（sop_projects/{name}/config.json）的讀寫、驗證與目錄管理。

schema_version 1：步驟用固定矩形 roi（舊式）。載入時每個 roi 會自動包成名為 "roi_<step_index>" 的固定區域。
schema_version 2：新增 anchors（會動的物件錨點）、zones（相對錨點或固定的區域）、步驟三種 type：
  hand_dwell      手在 zone 內停留 dwell_time_sec
  object_in_zone  object 錨點的框落在 zone 內（面積比例 ≥ min_overlap）維持 dwell_time_sec
  gesture         手勢分類器對 label 的機率 ≥ min_prob 維持 dwell_time_sec
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass, field

from scene import AnchorSpec, ZoneSpec

PROJECTS_DIR_NAME = "sop_projects"
CONFIG_FILE_NAME = "config.json"
TEMPLATES_DIR_NAME = "templates"
ERROR_LOGS_DIR_NAME = "error_logs"
INVALID_NAME_CHARS = set('\\/:*?"<>|')
TRIGGER_MODES = ("hand_only", "hand_and_visual")
STEP_TYPES = ("hand_dwell", "object_in_zone", "gesture", "box_state", "item", "hand_on_object")
END_TYPES = ("object_absent", "hands_absent", "object_in_zone", "hand_dwell", "gesture", "box_state")
ANCHOR_PICKS = ("largest", "all", "topmost", "bottommost")
COUNT_MODES = ("any", "all")
SUPPORTED_SCHEMAS = (1, 2)


class ProjectConfigError(Exception):
    """config.json 內容不合規格。訊息為繁體中文。"""


@dataclass
class Step:
    step_index: int
    step_name: str
    dwell_time_sec: float
    timeout_sec: float
    type: str = "hand_dwell"
    zone: str | None = None
    roi: list[int] | None = None            # v1 固定框；v2 不用
    trigger_mode: str = "hand_only"
    visual_match_min: float | None = None
    template_file: str = ""
    frame_number: int = 0
    object: str | None = None               # object_in_zone 用
    min_overlap: float = 0.5
    count: str = "any"
    label: str | None = None                # gesture 用
    min_prob: float = 0.6
    required: bool = True                   # 清單模式：這一步沒做要不要報
    continuous: bool = False                # 清單模式：dwell 要「連續」停留（中斷超過 0.3 秒就重算），不是整輪累加；
                                            # hand_on_object 用：手經過鞋子的零碎片段整輪加起來也會超過門檻


@dataclass
class Project:
    project_name: str
    resolution: list[int]
    fps: float
    steps: list[Step] = field(default_factory=list)
    video_file: str = "demo.avi"
    skip_alert_sec: float = 1.0
    cycle_arm_clear_sec: float = 0.5
    reset_zone: list[int] | None = None
    reset_dwell_sec: float = 1.0
    schema_version: int = 1
    anchors: list[AnchorSpec] = field(default_factory=list)
    zones: list[ZoneSpec] = field(default_factory=list)
    work_area: list[int] | None = None
    detect_every_n_frames: int = 15
    gesture_model: str | None = None        # 相對專案資料夾的 .pkl
    gesture_anchors: list[str] = field(default_factory=list)   # 手勢特徵的參考錨點，順序要與訓練時一致（例如 ["shoes", "box"]）
    cycle_end: Step | None = None           # 清單模式：一輪結束的條件（step_index = -1）；None = 依序模式
    cycle_timeout_sec: float | None = None  # 清單模式：一輪超過這麼久沒結束就報超時
    cycle_min_sec: float = 0.0              # 清單模式：一輪不到這麼久就結束 → 視為不是一輪，直接丟掉不報
    state_anchor: str | None = None         # 盒內狀態辨識：用哪個錨點的框裁盒內（例如 "box"）
    state_rel: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])   # 裁盒內的相對區域
    state_fixed: list[int] | None = None    # 工位固定時鎖定的固定區域 [x, y, w, h]；有設就優先用
    state_every_n_frames: int = 5           # 幾格判一次狀態（5：放完東西到下一個動作常只有 0.7 秒空檔，10 格一次投票來不及）
    dir_path: str = ""

    @property
    def mode(self) -> str:
        return "checklist" if self.cycle_end is not None else "ordered"

    @property
    def needs_hands(self) -> bool:
        """有沒有任何規則用到手部關鍵點：hand_dwell／gesture／hand_on_object／hands_absent 的步驟或結束條件、重置區、
        跟著盒子裁的盒內狀態／物品（手在盒內不判）。純物件版、固定桌面範圍又沒有手部步驟的專案用不到，
        監控迴圈就跳過 MediaPipe（每格 50～60 ms）。"""
        hand_types = ("hand_dwell", "gesture", "hand_on_object", "hands_absent")
        if any(s.type in hand_types for s in self.steps) or self.reset_zone or self.gesture_model:
            return True
        if self.cycle_end is not None and self.cycle_end.type in hand_types + ("box_state",):
            return True
        box_mode = self.state_fixed is None and self.state_anchor is not None
        return box_mode and any(s.type in ("box_state", "item") for s in self.steps)

    @property
    def video_path(self) -> str:
        return os.path.join(self.dir_path, self.video_file)

    def zone_specs(self) -> dict[str, ZoneSpec]:
        return {z.name: z for z in self.zones}


# ---------------------------------------------------------------- 路徑

def app_base_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def projects_dir() -> str:
    path = os.path.join(app_base_dir(), PROJECTS_DIR_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def list_projects() -> list[str]:
    base = projects_dir()
    return [n for n in sorted(os.listdir(base)) if os.path.isfile(os.path.join(base, n, CONFIG_FILE_NAME))]


def validate_project_name(name: str) -> None:
    if not isinstance(name, str) or not (1 <= len(name) <= 50):
        raise ProjectConfigError("專案名稱長度必須是 1～50 個字元")
    bad = sorted(set(name) & INVALID_NAME_CHARS)
    if bad:
        raise ProjectConfigError("專案名稱不可包含這些字元：" + " ".join(bad))
    if name != name.strip() or name.startswith(".") or name.endswith("."):
        raise ProjectConfigError("專案名稱頭尾不可是空白或句點")


# ---------------------------------------------------------------- 幾何

def rects_intersect(a, b) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def rect_inside(rect, width: int, height: int) -> bool:
    x, y, w, h = rect
    return x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= width and y + h <= height


# ---------------------------------------------------------------- 轉換

def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ProjectConfigError(msg)


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_rect(v) -> bool:
    return isinstance(v, list) and len(v) == 4 and all(_is_num(n) for n in v)


def _parse_anchors(raw) -> list[AnchorSpec]:
    _require(isinstance(raw, list), "anchors 必須是陣列")
    out: list[AnchorSpec] = []
    names = set()
    for i, a in enumerate(raw):
        _require(isinstance(a, dict), f"anchors[{i}] 必須是物件")
        _require(isinstance(a.get("name"), str) and a["name"], f"anchors[{i}].name 必須是非空字串")
        _require(a["name"] not in names, f"anchors 名稱重複：{a['name']}")
        names.add(a["name"])
        _require(isinstance(a.get("prompt"), str) and a["prompt"], f"anchors[{i}].prompt 必須是非空字串")
        pick = a.get("pick", "largest")
        _require(pick in ANCHOR_PICKS, f"anchors[{i}].pick 只能是 {ANCHOR_PICKS}")
        mc = a.get("max_count", 1)
        _require(isinstance(mc, int) and mc >= 1, f"anchors[{i}].max_count 必須是 ≥1 的整數")
        exc = a.get("exclude_containing", [])
        _require(isinstance(exc, list) and all(isinstance(s, str) for s in exc), f"anchors[{i}].exclude_containing 必須是名稱陣列")
        out.append(AnchorSpec(name=a["name"], prompt=a["prompt"], min_conf=float(a.get("min_conf", 0.25)),
                              pick=pick, max_count=mc, min_area=float(a.get("min_area", 0.0)),
                              max_age_sec=float(a.get("max_age_sec", 2.0)), exclude_containing=list(exc)))
    for a in out:
        for other in a.exclude_containing:
            _require(other in names and other != a.name, f"anchors「{a.name}」.exclude_containing 的「{other}」不是其他錨點的名稱")
    return out


def _parse_zones(raw, anchor_names: set[str], res) -> list[ZoneSpec]:
    _require(isinstance(raw, list), "zones 必須是陣列")
    out: list[ZoneSpec] = []
    names = set()
    for i, z in enumerate(raw):
        _require(isinstance(z, dict), f"zones[{i}] 必須是物件")
        _require(isinstance(z.get("name"), str) and z["name"], f"zones[{i}].name 必須是非空字串")
        _require(z["name"] not in names, f"zones 名稱重複：{z['name']}")
        names.add(z["name"])
        anchor = z.get("anchor")
        fixed = z.get("fixed")
        if anchor is None:
            _require(_is_rect(fixed), f"zones[{i}] 沒有 anchor 就必須給 fixed [x, y, w, h]")
            fixed = [int(v) for v in fixed]
            _require(rect_inside(fixed, res[0], res[1]), f"zones[{i}].fixed 超出 resolution 範圍")
        else:
            _require(anchor in anchor_names, f"zones[{i}].anchor「{anchor}」不存在於 anchors")
            fixed = None
        rel = z.get("rel", [0.0, 0.0, 1.0, 1.0])
        _require(_is_rect(rel) and rel[2] > 0 and rel[3] > 0, f"zones[{i}].rel 必須是 [rx, ry, rw, rh] 且寬高 > 0")
        idx = z.get("index")
        _require(idx is None or (isinstance(idx, int) and idx >= 0), f"zones[{i}].index 必須是 null 或非負整數")
        out.append(ZoneSpec(name=z["name"], anchor=anchor, rel=[float(v) for v in rel], index=idx, fixed=fixed))
    return out


def _parse_condition_fields(s: dict, where: str, stype: str, step: Step, zones: list[ZoneSpec], anchor_names: set[str],
                            gesture_model) -> None:
    """hand_dwell / object_in_zone / gesture / object_absent 共用的欄位檢查。"""
    if stype in ("hand_dwell", "object_in_zone"):
        z = s.get("zone")
        _require(isinstance(z, str) and any(zz.name == z for zz in zones), f"{where}.zone「{z}」不存在於 zones")
        step.zone = z
    if stype in ("object_in_zone", "object_absent", "hand_on_object"):
        obj = s.get("object")
        _require(isinstance(obj, str) and obj in anchor_names, f"{where}.object「{obj}」不存在於 anchors")
        step.object = obj
    if stype == "object_in_zone":
        mo = s.get("min_overlap", 0.5)
        _require(_is_num(mo) and 0.0 < mo <= 1.0, f"{where}.min_overlap 必須在 0～1")
        step.min_overlap = float(mo)
        cnt = s.get("count", "any")
        _require(cnt in COUNT_MODES, f"{where}.count 只能是 any 或 all")
        step.count = cnt
    if stype == "gesture":
        _require(isinstance(s.get("label"), str) and s["label"], f"{where}.label 必須是非空字串")
        _require(gesture_model is not None, f"{where} 是 gesture 類型，但專案沒有設定 gesture_model")
        step.label = s["label"]
        mp_ = s.get("min_prob", 0.6)
        _require(_is_num(mp_) and 0.0 < mp_ <= 1.0, f"{where}.min_prob 必須在 0～1")
        step.min_prob = float(mp_)
    if stype == "box_state":
        _require(isinstance(s.get("label"), str) and s["label"], f"{where}.label（狀態名稱）必須是非空字串")
        step.label = s["label"]
    if stype == "item":
        _require(isinstance(s.get("label"), str) and s["label"], f"{where}.label（物品名稱）必須是非空字串")
        step.label = s["label"]


def _parse_cycle_end(raw, zones, anchor_names, gesture_model) -> Step | None:
    if raw is None:
        return None
    _require(isinstance(raw, dict), "cycle_end 必須是物件或 null")
    etype = raw.get("type")
    _require(etype in END_TYPES, f"cycle_end.type 只能是 {END_TYPES}")
    hold = raw.get("hold_sec", 2.0)
    _require(_is_num(hold) and hold > 0, "cycle_end.hold_sec 必須大於 0")
    end = Step(step_index=-1, step_name="cycle_end", dwell_time_sec=float(hold), timeout_sec=1.0, type=etype)
    _parse_condition_fields(raw, "cycle_end", etype, end, zones, anchor_names, gesture_model)
    return end


def project_from_dict(data: dict, dir_path: str = "") -> Project:
    _require(isinstance(data, dict), "config.json 最外層必須是物件")
    schema = data.get("schema_version")
    _require(schema in SUPPORTED_SCHEMAS, f"schema_version 必須是 {SUPPORTED_SCHEMAS} 之一")
    _require(isinstance(data.get("project_name"), str) and data["project_name"], "project_name 必須是非空字串")
    res = data.get("resolution")
    _require(isinstance(res, list) and len(res) == 2 and all(isinstance(n, int) and n > 0 for n in res),
             "resolution 必須是 [寬, 高] 兩個正整數")
    _require(_is_num(data.get("fps")) and data["fps"] > 0, "fps 必須是大於 0 的數字")
    video_file = data.get("video_file", "demo.avi")
    _require(isinstance(video_file, str) and video_file, "video_file 必須是非空字串")
    skip = data.get("skip_alert_sec", 1.0)
    _require(_is_num(skip) and skip > 0, "skip_alert_sec 必須大於 0")
    arm = data.get("cycle_arm_clear_sec", 0.5)
    _require(_is_num(arm) and arm >= 0, "cycle_arm_clear_sec 不可為負數")
    reset_dwell = data.get("reset_dwell_sec", 1.0)
    _require(_is_num(reset_dwell) and reset_dwell > 0, "reset_dwell_sec 必須大於 0")
    reset_zone = data.get("reset_zone")
    if reset_zone is not None:
        _require(_is_rect(reset_zone), "reset_zone 必須是 [x, y, w, h] 或 null")
        reset_zone = [int(v) for v in reset_zone]
        _require(rect_inside(reset_zone, res[0], res[1]), "reset_zone 超出 resolution 範圍")
    work_area = data.get("work_area")
    if work_area is not None:
        _require(_is_rect(work_area), "work_area 必須是 [x, y, w, h] 或 null")
        work_area = [int(v) for v in work_area]
    every = data.get("detect_every_n_frames", 15)
    _require(isinstance(every, int) and every >= 1, "detect_every_n_frames 必須是 ≥1 的整數")
    gesture_model = data.get("gesture_model")
    _require(gesture_model is None or (isinstance(gesture_model, str) and gesture_model), "gesture_model 必須是 null 或檔名")
    ga = data.get("gesture_anchors", data.get("gesture_anchor"))
    if ga is None:
        gesture_anchors: list[str] = []
    elif isinstance(ga, str):
        gesture_anchors = [ga]
    else:
        _require(isinstance(ga, list) and all(isinstance(s, str) for s in ga), "gesture_anchors 必須是名稱陣列")
        gesture_anchors = list(ga)

    anchors = _parse_anchors(data.get("anchors", [])) if schema == 2 else []
    anchor_names = {a.name for a in anchors}
    zones = _parse_zones(data.get("zones", []), anchor_names, res) if schema == 2 else []
    for g in gesture_anchors:
        _require(g in anchor_names, f"gesture_anchors「{g}」不存在於 anchors")

    raw_steps = data.get("steps")
    _require(isinstance(raw_steps, list), "steps 必須是陣列")
    steps: list[Step] = []
    for pos, s in enumerate(raw_steps):
        _require(isinstance(s, dict), f"steps[{pos}] 必須是物件")
        _require(s.get("step_index") == pos, f"steps[{pos}].step_index 必須等於 {pos}")
        _require(isinstance(s.get("step_name"), str) and s["step_name"], f"steps[{pos}].step_name 必須是非空字串")
        _require(_is_num(s.get("dwell_time_sec")) and s["dwell_time_sec"] > 0, f"steps[{pos}].dwell_time_sec 必須大於 0")
        _require(_is_num(s.get("timeout_sec")) and s["timeout_sec"] > 0, f"steps[{pos}].timeout_sec 必須大於 0")
        stype = s.get("type", "hand_dwell")
        _require(stype in STEP_TYPES, f"steps[{pos}].type 只能是 {STEP_TYPES}")
        mode = s.get("trigger_mode", "hand_only")
        _require(mode in TRIGGER_MODES, f"steps[{pos}].trigger_mode 只能是 hand_only 或 hand_and_visual")
        vmm = s.get("visual_match_min")
        if mode == "hand_and_visual":
            _require(_is_num(vmm) and 0.0 <= vmm <= 1.0, f"steps[{pos}].visual_match_min 必須是 0～1 的數字（hand_and_visual 必填）")
        else:
            _require(vmm is None or (_is_num(vmm) and 0.0 <= vmm <= 1.0), f"steps[{pos}].visual_match_min 必須是 null 或 0～1 的數字")
        step = Step(step_index=pos, step_name=s["step_name"], dwell_time_sec=float(s["dwell_time_sec"]),
                    timeout_sec=float(s["timeout_sec"]), type=stype, trigger_mode=mode,
                    visual_match_min=None if vmm is None else float(vmm),
                    template_file=s.get("template_file", "") or "", frame_number=int(s.get("frame_number", 0) or 0))
        if schema == 1:
            _require(_is_rect(s.get("roi")), f"steps[{pos}].roi 必須是 [x, y, w, h]")
            roi = [int(v) for v in s["roi"]]
            _require(rect_inside(roi, res[0], res[1]), f"steps[{pos}].roi 超出 resolution 範圍或寬高不是正數")
            _require(isinstance(s.get("template_file"), str) and s["template_file"], f"steps[{pos}].template_file 必須是非空字串")
            step.roi = roi
            zname = f"roi_{pos}"
            zones.append(ZoneSpec(name=zname, anchor=None, fixed=roi))
            step.zone = zname
        else:
            _parse_condition_fields(s, f"steps[{pos}]", stype, step, zones, anchor_names, gesture_model)
            _require(mode == "hand_only" or stype == "hand_dwell", f"steps[{pos}] 只有 hand_dwell 可以用 hand_and_visual")
            req = s.get("required", True)
            _require(isinstance(req, bool), f"steps[{pos}].required 必須是 true/false")
            step.required = req
            cont = s.get("continuous", False)
            _require(isinstance(cont, bool), f"steps[{pos}].continuous 必須是 true/false")
            step.continuous = cont
        steps.append(step)

    cycle_end = _parse_cycle_end(data.get("cycle_end"), zones, anchor_names, gesture_model) if schema == 2 else None
    cycle_timeout = data.get("cycle_timeout_sec")
    _require(cycle_timeout is None or (_is_num(cycle_timeout) and cycle_timeout > 0), "cycle_timeout_sec 必須是 null 或大於 0")
    cycle_min = data.get("cycle_min_sec", 0.0)
    _require(_is_num(cycle_min) and cycle_min >= 0, "cycle_min_sec 不可為負數")
    state_anchor = data.get("state_anchor")
    _require(state_anchor is None or state_anchor in anchor_names, f"state_anchor「{state_anchor}」不存在於 anchors")
    state_rel = data.get("state_rel", [0.0, 0.0, 1.0, 1.0])
    _require(_is_rect(state_rel) and state_rel[2] > 0 and state_rel[3] > 0, "state_rel 必須是 [rx, ry, rw, rh]")
    state_fixed = data.get("state_fixed")
    if state_fixed is not None:
        _require(_is_rect(state_fixed), "state_fixed 必須是 [x, y, w, h] 或 null")
        state_fixed = [int(v) for v in state_fixed]
        _require(rect_inside(state_fixed, res[0], res[1]), "state_fixed 超出 resolution 範圍")
    state_every = data.get("state_every_n_frames", 5)
    _require(isinstance(state_every, int) and state_every >= 1, "state_every_n_frames 必須是 ≥1 的整數")
    if any(s.get("type") in ("box_state", "item") for s in raw_steps if isinstance(s, dict)) or (
            isinstance(data.get("cycle_end"), dict) and data["cycle_end"].get("type") == "box_state"):
        _require(state_anchor is not None or state_fixed is not None, "有 box_state／item 步驟時，要設定 state_anchor（錨點）或 state_fixed（固定區域）")

    # 固定區域之間不可重疊（會動的區域無法在載入時檢查）
    zmap = {z.name: z for z in zones}
    fixed_steps = [(st, zmap[st.zone].fixed) for st in steps if st.zone and zmap[st.zone].fixed]
    for i in range(len(fixed_steps)):
        for j in range(i + 1, len(fixed_steps)):
            si, ri = fixed_steps[i]; sj, rj = fixed_steps[j]
            _require(not rects_intersect(ri, rj), f"步驟 {si.step_index}「{si.step_name}」與步驟 {sj.step_index}「{sj.step_name}」的 ROI 重疊")
        if reset_zone is not None:
            si, ri = fixed_steps[i]
            _require(not rects_intersect(ri, reset_zone), f"reset_zone 與步驟 {si.step_index}「{si.step_name}」的 ROI 重疊")

    return Project(
        project_name=data["project_name"], resolution=[int(res[0]), int(res[1])], fps=float(data["fps"]),
        steps=steps, video_file=video_file, skip_alert_sec=float(skip), cycle_arm_clear_sec=float(arm),
        reset_zone=reset_zone, reset_dwell_sec=float(reset_dwell), schema_version=int(schema),
        anchors=anchors, zones=zones, work_area=work_area, detect_every_n_frames=int(every),
        gesture_model=gesture_model, gesture_anchors=gesture_anchors,
        cycle_end=cycle_end, cycle_timeout_sec=None if cycle_timeout is None else float(cycle_timeout),
        cycle_min_sec=float(cycle_min), state_anchor=state_anchor, state_rel=[float(v) for v in state_rel],
        state_fixed=state_fixed, state_every_n_frames=int(state_every), dir_path=dir_path,
    )


def project_to_dict(project: Project) -> dict:
    d = {
        "schema_version": project.schema_version,
        "project_name": project.project_name,
        "video_file": project.video_file,
        "resolution": list(project.resolution),
        "fps": project.fps,
        "skip_alert_sec": project.skip_alert_sec,
        "cycle_arm_clear_sec": project.cycle_arm_clear_sec,
        "reset_zone": None if project.reset_zone is None else list(project.reset_zone),
        "reset_dwell_sec": project.reset_dwell_sec,
    }
    if project.schema_version == 1:
        d["steps"] = [
            {"step_index": s.step_index, "step_name": s.step_name, "frame_number": s.frame_number,
             "roi": list(s.roi) if s.roi else None, "trigger_mode": s.trigger_mode, "dwell_time_sec": s.dwell_time_sec,
             "timeout_sec": s.timeout_sec, "visual_match_min": s.visual_match_min, "template_file": s.template_file}
            for s in project.steps
        ]
        return d
    d.update({
        "work_area": None if project.work_area is None else list(project.work_area),
        "detect_every_n_frames": project.detect_every_n_frames,
        "gesture_model": project.gesture_model,
        "gesture_anchors": list(project.gesture_anchors),
        "anchors": [{"name": a.name, "prompt": a.prompt, "min_conf": a.min_conf, "pick": a.pick, "max_count": a.max_count,
                     "min_area": a.min_area, "max_age_sec": a.max_age_sec, "exclude_containing": list(a.exclude_containing)}
                    for a in project.anchors],
        "zones": [{"name": z.name, "anchor": z.anchor, "rel": list(z.rel), "index": z.index, "fixed": z.fixed} for z in project.zones],
        "cycle_timeout_sec": project.cycle_timeout_sec,
        "cycle_min_sec": project.cycle_min_sec,
        "state_anchor": project.state_anchor,
        "state_rel": list(project.state_rel),
        "state_fixed": None if project.state_fixed is None else list(project.state_fixed),
        "state_every_n_frames": project.state_every_n_frames,
        "cycle_end": None,
        "steps": [],
    })
    if project.cycle_end is not None:
        e = project.cycle_end
        row = {"type": e.type, "hold_sec": e.dwell_time_sec}
        if e.type in ("hand_dwell", "object_in_zone"):
            row["zone"] = e.zone
        if e.type in ("object_in_zone", "object_absent"):
            row["object"] = e.object
        # hands_absent 沒有額外欄位
        if e.type == "object_in_zone":
            row.update({"min_overlap": e.min_overlap, "count": e.count})
        if e.type == "gesture":
            row.update({"label": e.label, "min_prob": e.min_prob})
        if e.type in ("box_state", "item"):
            row["label"] = e.label
        d["cycle_end"] = row
    for s in project.steps:
        row = {"step_index": s.step_index, "step_name": s.step_name, "type": s.type, "dwell_time_sec": s.dwell_time_sec,
               "timeout_sec": s.timeout_sec, "required": s.required, "trigger_mode": s.trigger_mode,
               "visual_match_min": s.visual_match_min, "template_file": s.template_file, "frame_number": s.frame_number,
               "continuous": s.continuous}
        if s.type in ("hand_dwell", "object_in_zone"):
            row["zone"] = s.zone
        if s.type == "object_in_zone":
            row.update({"object": s.object, "min_overlap": s.min_overlap, "count": s.count})
        if s.type == "hand_on_object":
            row["object"] = s.object
        if s.type == "gesture":
            row.update({"label": s.label, "min_prob": s.min_prob})
        if s.type in ("box_state", "item"):
            row["label"] = s.label
        d["steps"].append(row)
    return d


# ---------------------------------------------------------------- 讀寫

def _resolve_project_dir(name_or_path: str) -> str:
    if os.path.isdir(name_or_path):
        return os.path.abspath(name_or_path)
    return os.path.join(projects_dir(), name_or_path)


def _ensure_templates(project: Project) -> None:
    """hand_and_visual 步驟的 template 檔不存在而示範影片存在時，從 frame_number 自動裁切補上（只支援固定區域）。"""
    zmap = project.zone_specs()
    missing = [s for s in project.steps
               if s.template_file and s.zone and zmap[s.zone].fixed
               and not os.path.isfile(os.path.join(project.dir_path, s.template_file))]
    if not missing or not os.path.isfile(project.video_path):
        return
    from video_recorder import FrameReader, write_image  # 延後匯入，避免循環相依
    reader = FrameReader(project.video_path)
    try:
        for s in missing:
            frame = reader.get_frame(min(s.frame_number, max(reader.frame_count - 1, 0)))
            x, y, w, h = zmap[s.zone].fixed
            crop = frame[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            path = os.path.join(project.dir_path, s.template_file)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            write_image(path, crop)
    finally:
        reader.release()


def load_project(name_or_path: str) -> Project:
    dir_path = _resolve_project_dir(name_or_path)
    config_path = os.path.join(dir_path, CONFIG_FILE_NAME)
    if not os.path.isfile(config_path):
        raise ProjectConfigError(f"找不到設定檔：{config_path}")
    try:
        with open(config_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ProjectConfigError(f"config.json 不是合法的 JSON：{e}") from e
    project = project_from_dict(data, dir_path)
    _ensure_templates(project)
    for s in project.steps:
        if s.trigger_mode == "hand_and_visual":
            tpl = os.path.join(dir_path, s.template_file)
            if not os.path.isfile(tpl):
                raise ProjectConfigError(f"步驟 {s.step_index}「{s.step_name}」是 hand_and_visual，但找不到樣板圖：{tpl}")
    if project.gesture_model and not os.path.isfile(os.path.join(dir_path, project.gesture_model)):
        raise ProjectConfigError(f"找不到手勢模型：{os.path.join(dir_path, project.gesture_model)}")
    return project


def save_project(project: Project) -> None:
    if not project.dir_path:
        project.dir_path = os.path.join(projects_dir(), project.project_name)
    os.makedirs(os.path.join(project.dir_path, TEMPLATES_DIR_NAME), exist_ok=True)
    os.makedirs(os.path.join(project.dir_path, ERROR_LOGS_DIR_NAME), exist_ok=True)
    tmp = os.path.join(project.dir_path, CONFIG_FILE_NAME + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(project_to_dict(project), f, ensure_ascii=False, indent=2)
    os.replace(tmp, os.path.join(project.dir_path, CONFIG_FILE_NAME))


def create_project(name: str, resolution=(1280, 720), fps: float = 30.0, schema_version: int = 2) -> Project:
    validate_project_name(name)
    dir_path = os.path.join(projects_dir(), name)
    if os.path.exists(dir_path):
        raise ProjectConfigError(f"專案「{name}」已存在")
    project = Project(project_name=name, resolution=[int(resolution[0]), int(resolution[1])], fps=float(fps),
                      schema_version=schema_version, dir_path=dir_path)
    save_project(project)
    return project


def delete_project(name: str) -> None:
    validate_project_name(name)
    dir_path = os.path.join(projects_dir(), name)
    if not os.path.isdir(dir_path):
        raise ProjectConfigError(f"專案「{name}」不存在")
    shutil.rmtree(dir_path)
