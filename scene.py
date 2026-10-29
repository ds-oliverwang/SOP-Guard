"""場景：物件錨點（由物件偵測器提供、會移動的框）與相對於錨點定義的區域。

- AnchorSpec：一種要追的物件（例如 "cardboard box" → 名稱 "box"），含挑選規則。
- SceneTracker：收偵測結果、依規則挑出每個錨點目前的框；偵測器不是每格都跑，沒新結果時沿用上一次（有效期 max_age_sec）。
- ZoneSpec / resolve_zone：區域可以是固定矩形（舊式 roi），或錨點框的相對子矩形（rel = [rx, ry, rw, rh]，0～1 的比例）。
所有矩形一律 [x, y, w, h]（像素，轉正後的畫面座標）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

Rect = tuple[float, float, float, float]


@dataclass
class Detection:
    label: str
    conf: float
    xyxy: tuple[float, float, float, float]

    @property
    def rect(self) -> Rect:
        x1, y1, x2, y2 = self.xyxy
        return (x1, y1, x2 - x1, y2 - y1)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class AnchorSpec:
    name: str
    prompt: str
    min_conf: float = 0.25
    pick: str = "largest"          # largest | all | topmost | bottommost
    max_count: int = 1
    min_area: float = 0.0
    max_age_sec: float = 2.0       # 超過這麼久沒再偵測到就視為消失
    exclude_containing: list[str] = field(default_factory=list)   # 候選框若「包住」這些錨點的框（面積比 ≥ 0.7）就不算
    # 例：鞋盒錨點設 ["shoes"]，可排除「鞋子底下那片紙板」被當成鞋盒


@dataclass
class ZoneSpec:
    name: str
    anchor: str | None = None      # None → 用 fixed
    rel: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])
    index: int | None = None       # 錨點有多個框時只取第幾個（依 pick 排序）；None = 全部
    fixed: list[float] | None = None


def point_in_rect(pt, rect: Rect) -> bool:
    x, y, w, h = rect
    px, py = pt
    return x <= px <= x + w and y <= py <= y + h


def rect_overlap_ratio(inner: Rect, outer: Rect) -> float:
    """inner 落在 outer 內的面積比例（0～1）。"""
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    if iw <= 0 or ih <= 0:
        return 0.0
    x1 = max(ix, ox); y1 = max(iy, oy)
    x2 = min(ix + iw, ox + ow); y2 = min(iy + ih, oy + oh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1) / (iw * ih)


def sub_rect(base: Rect, rel) -> Rect:
    x, y, w, h = base
    rx, ry, rw, rh = rel
    return (x + w * rx, y + h * ry, w * rw, h * rh)


class SceneTracker:
    def __init__(self, anchors: list[AnchorSpec], work_area: Rect | None = None):
        self.specs = {a.name: a for a in anchors}
        self.work_area = work_area
        self._boxes: dict[str, list[Rect]] = {a.name: [] for a in anchors}
        self._seen_at: dict[str, float] = {}

    def _candidates(self, dets: list[Detection], spec: AnchorSpec) -> list[Detection]:
        cands = [d for d in dets if d.label == spec.prompt and d.conf >= spec.min_conf and d.area >= spec.min_area]
        if self.work_area is not None:
            cands = [d for d in cands if point_in_rect(d.center, self.work_area)]
        return cands

    def update_detections(self, dets: list[Detection], t_now: float) -> None:
        """收一次偵測結果（整張畫面）。沒偵測到的錨點保留舊框直到過期。"""
        raw = {name: self._candidates(dets, spec) for name, spec in self.specs.items()}
        for name, spec in self.specs.items():
            cands = raw[name]
            for other in spec.exclude_containing:
                inner = [d.rect for d in raw.get(other, [])]
                cands = [d for d in cands if not any(rect_overlap_ratio(r, d.rect) >= 0.7 for r in inner)]
            if not cands:
                continue
            if spec.pick == "largest":
                cands.sort(key=lambda d: -d.area)
            elif spec.pick == "topmost":
                cands.sort(key=lambda d: d.xyxy[1])
            elif spec.pick == "bottommost":
                cands.sort(key=lambda d: -d.xyxy[3])
            elif spec.pick == "all":
                cands.sort(key=lambda d: d.xyxy[1])   # 多個框時固定依 y 由上到下排，index 才有意義
            self._boxes[name] = [d.rect for d in cands[:spec.max_count]]
            self._seen_at[name] = t_now

    def age(self, name: str, t_now: float) -> float | None:
        """距離上一次「真的偵測到」這個錨點過了幾秒；從沒偵測到回 None。anchors_at 會沿用舊框到 max_age_sec，
        要分辨「現在真的看得到」和「只是沿用」就看這個。"""
        seen = self._seen_at.get(name)
        return None if seen is None else max(0.0, t_now - seen)

    def anchors_at(self, t_now: float) -> dict[str, list[Rect]]:
        out: dict[str, list[Rect]] = {}
        for name, spec in self.specs.items():
            seen = self._seen_at.get(name)
            if seen is not None and (t_now - seen) <= spec.max_age_sec:
                out[name] = list(self._boxes[name])
            else:
                out[name] = []
        return out


def resolve_zone(spec: ZoneSpec, anchors: dict[str, list[Rect]]) -> list[Rect]:
    """回傳 0 到多個矩形；錨點不在畫面時回傳空 list（條件一律視為不成立）。"""
    if spec.anchor is None:
        return [tuple(spec.fixed)] if spec.fixed else []
    boxes = anchors.get(spec.anchor, [])
    if spec.index is not None:
        boxes = boxes[spec.index:spec.index + 1]
    return [sub_rect(b, spec.rel) for b in boxes]
