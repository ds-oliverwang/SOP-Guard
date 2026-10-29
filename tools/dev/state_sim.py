"""Pass 2: replay the recorded trace (YOLO + hands) and simulate region-selection rules / vote settings for the
box-state flow offline. CLIP embeddings are cached per (frame, rect) so re-running a variant is seconds."""
import os, sys, json, time, shutil, io
import numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import torch  # noqa
from project_store import load_project
from scene import Detection, SceneTracker, sub_rect, rect_overlap_ratio
from core_detector import SOPStateMachine, FrameContext, any_point_in_rect, crop_roi
from state_recognizer import StateLibrary, StateStream, Embedder
from video_recorder import VideoFileSource

D = os.path.join(ROOT, "cache", "_eval_innerbox")
TRACE = os.path.join(ROOT, "cache", "state_trace.json")
EMB_CACHE = os.path.join(ROOT, "cache", "state_trace_emb.npz")
SEGS = [("拿鞋盒",52,182),("開鞋盒",204,290),("放防黴片",325,522),("放鞋子",542,692),("拿鞋盒",1171,1249),("開鞋盒",1336,1466),
        ("放防黴片",1490,1651),("放鞋子",1671,1904),("拿鞋盒",2357,2528),("開鞋盒",2548,2662),("放防黴片",2687,2868),("放鞋子",2893,3085),
        ("拿鞋盒",3430,3616),("開鞋盒",3646,3724),("放防黴片",3739,3931),("放鞋子",3950,4106),("拿鞋盒",4461,4617),("開鞋盒",4652,4761),
        ("放防黴片",4796,4905),("放鞋子",4920,5101),("拿鞋盒",5477,5689),("開鞋盒",5709,5782),("放防黴片",5812,5937),("放鞋子",5962,6154)]
ROUNDS = [SEGS[i:i + 4] for i in range(0, len(SEGS), 4)]
EXAMPLE_ROUNDS = 3

p = load_project(D)
trace = json.load(open(TRACE, encoding="utf-8"))
rows = {r["f"]: r for r in trace["rows"]}
N = trace["frames"]; FPS = trace["fps"]
STEP = 5

# ---- frames (JPEG bytes in memory for every sampled frame)
print("loading frames...", flush=True)
src = VideoFileSource(p.video_path); jpg = {}; n = 0
while True:
    fr = src.read()
    if fr is None:
        break
    if n % STEP == 0:
        jpg[n] = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]
    n += 1
src.release()
def frame_at(f):
    return cv2.imdecode(jpg[f], cv2.IMREAD_COLOR)

# ---- scene replay: anchors at every sampled frame (same hold logic as the monitor)
tr = SceneTracker(p.anchors, tuple(p.work_area) if p.work_area else None)
anch = {}; hands = {}; last_hands = []
for f in range(N):
    r = rows.get(f)
    if r and "dets" in r:
        tr.update_detections([Detection(d[0], d[1], tuple(d[2:6])) for d in r["dets"]], f / FPS)
    if r and "hands" in r:
        last_hands = [tuple(h) for h in r["hands"]]
    if f % STEP == 0:
        anch[f] = tr.anchors_at(f / FPS); hands[f] = list(last_hands)

# ---- embeddings with cache
emb = Embedder(prefer_clip=True); assert emb.kind == "clip"
cache = {}
if os.path.isfile(EMB_CACHE):
    z = np.load(EMB_CACHE, allow_pickle=True); cache = {k: v for k, v in zip(z["keys"], z["vecs"])}
def key(f, rect):
    x, y, w, h = [int(round(v / 4) * 4) for v in rect]
    return f"{f}:{x},{y},{w},{h}"
def embed(f, rect):
    k = key(f, rect)
    if k not in cache:
        cache[k] = emb.embed(crop_roi(frame_at(f), rect)).astype(np.float16)
    return cache[k].astype(np.float32)
def save_cache():
    np.savez(EMB_CACHE, keys=np.array(list(cache.keys())), vecs=np.array(list(cache.values())))

# ---- region rules
def contains(a, b):   # a contains b
    return rect_overlap_ratio(b, a) >= 0.9
def choose_rect(rects, rule, typical=None, prefer=None, rel=p.state_rel):
    if not rects:
        return None
    if rule == "baseline":
        if prefer is not None and len(rects) > 1:
            def ov(r):
                a = sub_rect(r, rel); return min(rect_overlap_ratio(a, prefer), rect_overlap_ratio(prefer, a))
            rects = sorted(rects, key=lambda r: -ov(r))
        return sub_rect(rects[0], rel)
    # innermost: drop any rect that contains another candidate
    cands = [r for r in rects if not any(r is not o and contains(r, o) for o in rects)] or list(rects)
    if typical is not None and len(cands) > 1:
        tw, th = typical
        err = lambda r: max(abs(r[2] - tw) / tw, abs(r[3] - th) / th)
        close = [r for r in cands if err(r) <= 0.25]
        cands = close if close else sorted(cands, key=err)[:1]
    return sub_rect(cands[0], rel)

def expected_state(f):
    """ground truth-ish from SEGS: 未開 after grab until open end; 空盒 until card end; 已放防黴片 until shoe end; 已放鞋子 after."""
    st = None
    for grab, op, card, shoe in ROUNDS:
        if f < grab[1]: break
        if f < op[2]: st = "未開" if f < op[1] else "開盒中"
        elif f < card[2]: st = "空盒" if f < card[1] else "放片中"
        elif f < shoe[2]: st = "已放防黴片" if f < shoe[1] else "放鞋中"
        else: st = "已放鞋子/蓋上"
    return st

def build_library(rule, tmpdir):
    shutil.rmtree(tmpdir, ignore_errors=True); os.makedirs(tmpdir)
    lib = StateLibrary(tmpdir, embedder=emb)
    rects_used = []
    miss = 0
    for grab, op, card, shoe in ROUNDS[:EXAMPLE_ROUNDS]:
        ex = [("未開", grab[2] + 5), ("未開", op[1] - 5), ("空盒", op[2] + 40), ("空盒", op[2] + 70),
              ("已放防黴片", card[2] + 40), ("已放防黴片", card[2] + 70),
              ("已放鞋子", shoe[2] + 30), ("已放鞋子", shoe[2] + 60), ("已放鞋子", shoe[2] + 90)]
        for st, f in ex:
            f = int(round(f / STEP) * STEP)
            typical = None
            if rects_used:
                typical = (float(np.median([r[2] for r in rects_used])), float(np.median([r[3] for r in rects_used])))
            region = choose_rect(anch[f].get("box", []), rule, typical)
            if region is None:
                miss += 1; continue
            # which anchor rect produced it (for typical size)
            base = next((r for r in anch[f]["box"] if sub_rect(r, p.state_rel) == region), None)
            rects_used.append(base)
            lib.add_example(st, crop_roi(frame_at(f), region), f)
    typical = (float(np.median([r[2] for r in rects_used])), float(np.median([r[3] for r in rects_used]))) if rects_used else None
    # pre-fill lib vectors from our cache path (same embedder → same vectors)
    lib._ensure_vecs()
    return lib, typical, miss

def simulate(rule="innermost", every=5, votes=5, min_votes=3, max_d1=None, sticky_lost=False, margin=None, label=""):
    tmp = os.path.join(ROOT, "cache", "_sim_lib")
    lib, typical, miss = build_library(rule, tmp)
    loo = lib.leave_one_out()
    stream = StateStream(lib, votes=votes, min_votes=min_votes, min_margin=margin)
    fsm = SOPStateMachine(p, sound_enabled=False)
    region = None; moved = 0
    timeline = []   # (f, region?, hands_inside, raw, d1, margin, current)
    events_printed = 0; out = []
    for f in range(0, N, STEP):
        rects = anch[f].get("box", [])
        new = choose_rect(rects, rule, typical, prefer=region)
        if new is None:
            if not sticky_lost:
                region = None
        elif region is None:
            region = new
        else:
            same = min(rect_overlap_ratio(new, region), rect_overlap_ratio(region, new))
            if same < 0.6:
                moved += 1
                if moved >= 1:      # detections only change every 15 frames anyway
                    region = new; moved = 0
            else:
                moved = 0
        cur = stream.current; raw = None; d1 = None; mg = None; hin = None
        if f % every == 0:
            if region is None:
                cur = stream.push(None)
            else:
                hin = any_point_in_rect(hands[f], region)
                if hin:
                    cur = stream.push(None, hands_inside=True)
                else:
                    # recognise via cached embedding (replicates StateLibrary.recognize)
                    v = embed(f, region); X = np.array(lib._vecs)
                    d = np.linalg.norm(X - v, axis=1); best = {}
                    for lab, dist in zip(lib._labels, d):
                        best[lab] = min(best.get(lab, 9.0), float(dist))
                    ranked = sorted(best.items(), key=lambda kv: kv[1]); raw, d1 = ranked[0]
                    d2 = ranked[1][1] if len(ranked) > 1 else d1 + 1; mg = d2 - d1
                    mm = margin if margin is not None else 0.03
                    if (max_d1 is None or d1 <= max_d1) and mg >= mm:
                        stream.buf.append(raw)
                    if stream.buf:
                        top = max(set(stream.buf), key=stream.buf.count)
                        if stream.buf.count(top) >= min_votes:
                            stream.current = top
                    cur = stream.current
        timeline.append((f, region is not None, hin, raw, d1, mg, cur))
        ctx = FrameContext(hand_points=hands[f], anchors=anch[f], box_state=cur)
        fsm.update(None, ctx, f / FPS)
        if fsm.alert_active and (f / FPS - fsm.alert_started_at) >= 2.0:
            fsm.reset()
    evs = [(int(e["t"] * FPS), e["event_type"], e["step_index"], e["message"]) for e in fsm.events if e["event_type"] not in ("armed",)]
    print(f"\n=== {label or rule} every={every} votes={min_votes}/{votes} max_d1={max_d1} sticky_lost={sticky_lost} margin={margin}")
    print(f"examples={lib.counts()} miss={miss} typical={typical and tuple(int(v) for v in typical)} LOO={loo['accuracy']:.2f} {loo['confusions']}")
    for e in evs:
        print(f"  f={e[0]:5d} t={e[0]/FPS:6.1f}s {e[1]} {e[2]} {e[3]}  (expected: {expected_state(e[0])})")
    print(f"cycles={fsm.cycle_count} bad={fsm.bad_cycle_count} discarded={sum(1 for e in fsm.events if e['event_type'] == 'cycle_discarded')} "
          f"alerts={sum(1 for e in fsm.events if e['event_type'].startswith('alert_'))}")
    # per-expected-state accuracy of the stream (only frames with a clear expected state)
    agree = {}; tot = {}
    for f, has, hin, raw, d1, mg, cur in timeline:
        ex = expected_state(f)
        if ex in ("未開", "空盒", "已放防黴片") or ex == "已放鞋子/蓋上":
            k = ex; tot[k] = tot.get(k, 0) + 1
            if cur == (ex if ex != "已放鞋子/蓋上" else cur) and (cur in ("已放鞋子", "未開") if ex == "已放鞋子/蓋上" else True):
                agree[k] = agree.get(k, 0) + 1
    print("stream agreement:", {k: f"{agree.get(k, 0)}/{tot[k]}" for k in tot})
    return timeline, lib

if __name__ == "__main__":
    t0 = time.perf_counter()
    tl0, _ = simulate("baseline", every=10, label="baseline(現況)")
    save_cache()
    tl1, _ = simulate("innermost", every=5)
    save_cache()
    tl2, lib2 = simulate("innermost", every=5, sticky_lost=True)
    save_cache()
    # d1 distribution: frames where no box expected (between rounds) vs expected states, to pick max_d1
    print("\n--- d1 by expected state (innermost, sticky_lost) ---")
    byst = {}
    for f, has, hin, raw, d1, mg, cur in tl2:
        if d1 is None: continue
        byst.setdefault(expected_state(f), []).append(d1)
    for k, v in byst.items():
        v = np.array(v); print(f"  {k}: n={len(v)} d1 p10={np.percentile(v,10):.3f} p50={np.percentile(v,50):.3f} p90={np.percentile(v,90):.3f}")
    # raw-vs-expected confusion for the main states
    print("\n--- raw nearest state vs expected (innermost, sticky_lost, hands-free judgements) ---")
    conf = {}
    for f, has, hin, raw, d1, mg, cur in tl2:
        ex = expected_state(f)
        if raw is None or ex not in ("未開", "空盒", "已放防黴片", "已放鞋子/蓋上"): continue
        conf[(ex, raw)] = conf.get((ex, raw), 0) + 1
    for k in sorted(conf): print(f"  {k[0]:>10} -> {k[1]:<8} {conf[k]}")
    print(f"\n{time.perf_counter() - t0:.0f}s")
