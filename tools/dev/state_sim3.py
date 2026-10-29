"""Round 3: judge only while the box was really detected recently (not on the hold), drop the state when the box is gone,
closed-box examples also from after the lid goes on, shorter cycle_end hold."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import state_sim as S
import state_sim2 as S2
from scene import rect_overlap_ratio, point_in_rect
from core_detector import SOPStateMachine, FrameContext
from state_recognizer import StateStream

p, N, STEP, FPS, ROUNDS = S.p, S.N, S.STEP, S.FPS, S.ROUNDS
spec = p.anchors[0]
# last frame the box was really detected (same filters as SceneTracker._candidates)
def is_cand(d):
    label, conf, x1, y1, x2, y2 = d
    area = max(0, x2 - x1) * max(0, y2 - y1); cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return label == spec.prompt and conf >= spec.min_conf and area >= spec.min_area and (p.work_area is None or point_in_rect((cx, cy), tuple(p.work_area)))
last_seen = {}; seen = -10**9
for f in range(N):
    r = S.rows.get(f)
    if r and "dets" in r and any(is_cand(d) for d in r["dets"]):
        seen = f
    if f % STEP == 0:
        last_seen[f] = seen

def pick_examples_v3(gate, per_state=2, closed_after=True):
    ex = S2.pick_examples(gate, per_state)
    if closed_after:
        for grab, op, card, shoe in ROUNDS[:S.EXAMPLE_ROUNDS]:
            runs = S2.free_runs(shoe[2], shoe[2] + 90, gate)          # 蓋上之後、拿走之前
            flat = [f for r in runs for f in r if f >= shoe[2] + 40]
            ex += [("未開", f) for f in flat[:per_state]]
    return ex

def simulate(lib, gate, every=5, votes=5, min_votes=3, judge_within=30, gone_reset=True, end_hold=2.0, margin=0.03, label=""):
    p.cycle_end.dwell_time_sec = end_hold
    stream = StateStream(lib, votes=votes, min_votes=min_votes, min_margin=margin)
    fsm = SOPStateMachine(p, sound_enabled=False)
    region = None; X = np.array(lib._vecs); tl = []
    for f in range(0, N, STEP):
        new = S.choose_rect(S.anch[f].get("box", []), "innermost")
        if new is None:
            region = None
        elif region is None or min(rect_overlap_ratio(new, region), rect_overlap_ratio(region, new)) < 0.6:
            region = new
        raw = d1 = None; hin = None
        if region is None:
            if gone_reset:
                stream.reset()
            stream.push(None)
        elif f % every == 0:
            hin = gate(S.hands[f], region)
            fresh = (f - last_seen[f]) <= judge_within
            if hin or not fresh:
                stream.push(None, hands_inside=hin)
            else:
                v = S.embed(f, region); d = np.linalg.norm(X - v, axis=1); best = {}
                for lab, dist in zip(lib._labels, d):
                    best[lab] = min(best.get(lab, 9.0), float(dist))
                ranked = sorted(best.items(), key=lambda kv: kv[1]); raw, d1 = ranked[0]
                d2 = ranked[1][1] if len(ranked) > 1 else d1 + 1
                if d2 - d1 >= margin:
                    stream.buf.append(raw)
                if stream.buf:
                    top = max(set(stream.buf), key=stream.buf.count)
                    if stream.buf.count(top) >= min_votes:
                        stream.current = top
        cur = stream.current
        tl.append((f, region is not None, hin, raw, d1, None, cur))
        fsm.update(None, FrameContext(hand_points=S.hands[f], anchors=S.anch[f], box_state=cur), f / FPS)
        if fsm.alert_active and (f / FPS - fsm.alert_started_at) >= 2.0:
            fsm.reset()
    print(f"\n=== {label} every={every} votes={min_votes}/{votes} judge_within={judge_within} gone_reset={gone_reset} end_hold={end_hold}")
    for e in fsm.events:
        if e["event_type"] == "armed":
            continue
        f = int(e["t"] * FPS)
        print(f"  f={f:5d} t={f / FPS:6.1f}s {e['event_type']} {e['step_index']} {e['message']}  (expected: {S.expected_state(f)})")
    print(f"cycles={fsm.cycle_count} bad={fsm.bad_cycle_count} discarded={sum(1 for e in fsm.events if e['event_type'] == 'cycle_discarded')} "
          f"alerts={sum(1 for e in fsm.events if e['event_type'].startswith('alert_'))}")
    return tl

if __name__ == "__main__":
    tmp = os.path.join(S.ROOT, "cache", "_sim_lib3"); gate = S2.gate_inner
    lib, used = S2.build_library(S2.pick_examples(gate), tmp)
    print("##### examples v2:", lib.counts(), lib.leave_one_out())
    tl = simulate(lib, gate, label="v2 examples, fresh-only, gone-reset")
    S.save_cache()
    tl = simulate(lib, gate, judge_within=10**9, gone_reset=True, label="v2 examples, hold-judge, gone-reset")
    S.save_cache()
    lib, used = S2.build_library(pick_examples_v3(gate), tmp)
    print("\n##### examples v3 (+closed after lid):", lib.counts(), lib.leave_one_out())
    print("   frames:", used)
    tl = simulate(lib, gate, label="v3 examples, fresh-only, gone-reset")
    S.save_cache()
    tl = simulate(lib, gate, end_hold=1.0, label="v3 examples, fresh-only, gone-reset, end_hold=1")
    S.save_cache()
    S2.report_raw(tl, "v3 fresh-only")
    tl = simulate(lib, gate, votes=3, min_votes=2, end_hold=1.0, label="v3 examples, votes 2/3, end_hold=1")
    S.save_cache()
