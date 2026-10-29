"""Round 2 of the offline simulation: examples picked from real hands-free windows, hand gate variants, vote variants,
unknown-rejection threshold measured on 'box gone' frames."""
import os, sys, shutil
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import state_sim as S
from scene import sub_rect, rect_overlap_ratio
from core_detector import SOPStateMachine, FrameContext, crop_roi, point_in_rect
from state_recognizer import StateLibrary, StateStream

p, N, STEP, FPS, ROUNDS = S.p, S.N, S.STEP, S.FPS, S.ROUNDS

def gate_any(pts, region):
    return any(point_in_rect(q, region) for q in pts)
def gate_inner(pts, region, shrink=0.15, need=2):
    x, y, w, h = region
    inner = (x + w * shrink, y + h * shrink, w * (1 - 2 * shrink), h * (1 - 2 * shrink))
    for i in range(0, len(pts), 4):                      # 每手 4 點（拇指尖、食指尖、中指尖、掌心）
        if sum(point_in_rect(q, inner) for q in pts[i:i + 4]) >= need:
            return True
    return False

def free_runs(f0, f1, gate):
    """hands-free sampled frames in [f0, f1] grouped into consecutive runs (region must exist)."""
    runs = []; cur = []
    for f in range(int(np.ceil(f0 / STEP)) * STEP, f1 + 1, STEP):
        reg = S.choose_rect(S.anch[f].get("box", []), "innermost")
        ok = reg is not None and not gate(S.hands[f], reg)
        if ok:
            cur.append(f)
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    return runs

def pick_examples(gate, per_state=2):
    """未開: closed box before opening; 空盒: first free frames after open; 已放防黴片: last free frames before shoe;
    已放鞋子: middle of the longest free run during shoe placement (before the lid goes on)."""
    ex = []
    for grab, op, card, shoe in ROUNDS[:S.EXAMPLE_ROUNDS]:
        ex += [("未開", f) for f in [grab[2] + 5, op[1] - 5]]
        runs = free_runs(op[2], card[1] + 15, gate)
        flat = [f for r in runs for f in r]; ex += [("空盒", f) for f in flat[:per_state]]
        runs = free_runs(card[1], shoe[1], gate)
        flat = [f for r in runs for f in r]; ex += [("已放防黴片", f) for f in flat[-per_state:]]
        runs = free_runs(shoe[1], shoe[2] + 40, gate)
        if runs:
            r = max(runs, key=len); m = len(r) // 2
            ex += [("已放鞋子", f) for f in r[max(0, m - 1):m + per_state - 1]]
    return ex

def build_library(examples, tmpdir):
    shutil.rmtree(tmpdir, ignore_errors=True); os.makedirs(tmpdir)
    lib = StateLibrary(tmpdir, embedder=S.emb); used = []
    for st, f in examples:
        f = int(round(f / STEP) * STEP)
        region = S.choose_rect(S.anch[f].get("box", []), "innermost")
        if region is None:
            continue
        lib.add_example(st, crop_roi(S.frame_at(f), region), f); used.append((st, f))
    lib._ensure_vecs()
    return lib, used

def between_rounds(f):
    """frames where the box should be gone (after lid closed + 3 s hold, before next grab)."""
    for i, (grab, op, card, shoe) in enumerate(ROUNDS):
        nxt = ROUNDS[i + 1][0][1] if i + 1 < len(ROUNDS) else N
        if shoe[2] + 150 <= f < nxt - 10:
            return True
    return False

def simulate(lib, gate, every=5, votes=3, min_votes=2, max_d1=None, sticky_lost=False, margin=0.03, label=""):
    stream = StateStream(lib, votes=votes, min_votes=min_votes, min_margin=margin)
    fsm = SOPStateMachine(p, sound_enabled=False)
    region = None; tl = []; X = np.array(lib._vecs)
    for f in range(0, N, STEP):
        new = S.choose_rect(S.anch[f].get("box", []), "innermost")
        if new is None:
            if not sticky_lost:
                region = None
        elif region is None:
            region = new
        else:
            same = min(rect_overlap_ratio(new, region), rect_overlap_ratio(region, new))
            if same < 0.6:
                region = new
        raw = d1 = mg = None; hin = None
        if f % every == 0:
            if region is None:
                stream.push(None)
            else:
                hin = gate(S.hands[f], region)
                if hin:
                    stream.push(None, hands_inside=True)
                else:
                    v = S.embed(f, region); d = np.linalg.norm(X - v, axis=1); best = {}
                    for lab, dist in zip(lib._labels, d):
                        best[lab] = min(best.get(lab, 9.0), float(dist))
                    ranked = sorted(best.items(), key=lambda kv: kv[1]); raw, d1 = ranked[0]
                    d2 = ranked[1][1] if len(ranked) > 1 else d1 + 1; mg = d2 - d1
                    if (max_d1 is None or d1 <= max_d1) and mg >= margin:
                        stream.buf.append(raw)
                    if stream.buf:
                        top = max(set(stream.buf), key=stream.buf.count)
                        if stream.buf.count(top) >= min_votes:
                            stream.current = top
        cur = stream.current
        tl.append((f, region is not None, hin, raw, d1, mg, cur))
        fsm.update(None, FrameContext(hand_points=S.hands[f], anchors=S.anch[f], box_state=cur), f / FPS)
        if fsm.alert_active and (f / FPS - fsm.alert_started_at) >= 2.0:
            fsm.reset()
    print(f"\n=== {label} every={every} votes={min_votes}/{votes} max_d1={max_d1} sticky_lost={sticky_lost}")
    for e in fsm.events:
        if e["event_type"] == "armed":
            continue
        f = int(e["t"] * FPS)
        print(f"  f={f:5d} t={f / FPS:6.1f}s {e['event_type']} {e['step_index']} {e['message']}  (expected: {S.expected_state(f)})")
    print(f"cycles={fsm.cycle_count} bad={fsm.bad_cycle_count} discarded={sum(1 for e in fsm.events if e['event_type'] == 'cycle_discarded')} "
          f"alerts={sum(1 for e in fsm.events if e['event_type'].startswith('alert_'))}")
    return tl

def report_raw(tl, label):
    conf = {}; gone = []
    for f, has, hin, raw, d1, mg, cur in tl:
        if raw is None:
            continue
        if between_rounds(f):
            gone.append(d1); continue
        ex = S.expected_state(f)
        conf[(ex, raw)] = conf.get((ex, raw), 0) + 1
    print(f"--- raw nearest vs expected ({label}) ---")
    for k in sorted(conf, key=lambda k: (k[0] or "", k[1])):
        print(f"  {str(k[0]):>10} -> {k[1]:<8} {conf[k]}")
    if gone:
        g = np.array(gone); print(f"  box-gone frames judged: n={len(g)} d1 p10={np.percentile(g, 10):.3f} p50={np.percentile(g, 50):.3f} p90={np.percentile(g, 90):.3f}")
    byst = {}
    for f, has, hin, raw, d1, mg, cur in tl:
        if d1 is None or between_rounds(f):
            continue
        byst.setdefault(S.expected_state(f), []).append(d1)
    for k, v in byst.items():
        v = np.array(v); print(f"  {str(k):>10}: n={len(v):3d} d1 p50={np.percentile(v, 50):.3f} p90={np.percentile(v, 90):.3f} max={v.max():.3f}")

if __name__ == "__main__":
    tmp = os.path.join(S.ROOT, "cache", "_sim_lib2")
    for gate_name, gate in [("any", gate_any), ("inner", gate_inner)]:
        ex = pick_examples(gate)
        lib, used = build_library(ex, tmp)
        print(f"\n##### gate={gate_name} examples: {lib.counts()}  LOO={lib.leave_one_out()}")
        print("   frames:", used)
        tl = simulate(lib, gate, votes=3, min_votes=2, label=f"gate={gate_name}")
        S.save_cache()
        report_raw(tl, gate_name)
        tl = simulate(lib, gate, votes=5, min_votes=3, label=f"gate={gate_name}")
        S.save_cache()
    # unknown threshold: inner gate, sticky region when box lost, with/without max_d1
    ex = pick_examples(gate_inner); lib, used = build_library(ex, tmp)
    for md in (None, 0.45, 0.40, 0.35):
        tl = simulate(lib, gate_inner, votes=3, min_votes=2, max_d1=md, sticky_lost=True, label=f"inner sticky max_d1={md}")
        S.save_cache()
    report_raw(tl, "inner sticky")
