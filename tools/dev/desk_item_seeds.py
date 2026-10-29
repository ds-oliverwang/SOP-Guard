"""Write the experiment's seeds into the project's items/seeds.json (frame coordinates) and convert cache_trace.json into
items/trace.json, then run item_recognizer.build_dataset the way the UI will."""
import os, sys, json, shutil
sys.path.insert(0, os.path.dirname(__file__))
from item_core import *   # noqa: status, D, ROOT, REFS
import item_recognizer as ir
# seeds in crop coords (from item_dataset_build.py) → frame coords via the region of the nearest sampled frame
SEEDS = [("標籤", 1000, (255, 462, 340, 326)), ("黑色小物", 1000, (537, 470, 207, 217)), ("鞋子", 1100, (130, 80, 630, 600)),
         ("標籤", 2000, (382, 441, 180, 116)), ("黑色小物", 2000, (561, 441, 155, 162)), ("鞋子", 2120, (20, 80, 620, 700)),
         ("黑色小物", 2850, (290, 280, 195, 165)), ("標籤", 2850, (265, 444, 300, 164)), ("鞋子", 2960, (0, 72, 380, 497)),
         ("鞋子", 3850, (0, 42, 498, 493)),
         ("黑色小物", 4700, (232, 194, 155, 116)), ("標籤", 4700, (483, 272, 165, 135)), ("鞋子", 4865, (171, 92, 564, 596))]
seeds = ir.ItemSeeds(D); seeds.order, seeds.refs, seeds.seeds = [], [], []
for r in REFS:
    seeds.add_ref(r)
for name, f, (x, y, w, h) in SEEDS:
    f5 = f - f % STEP
    reg = status[f5][0] or status[f5 - 5][0]
    seeds.add_seed(name, f, (int(reg[0] + x), int(reg[1] + y), w, h))
print("seeds:", seeds.counts(), "refs:", seeds.refs)
# trace: reuse cache_trace.json
d = json.load(open(os.path.join(D, "cache_trace.json"), encoding="utf-8"))
d["fingerprint"] = ir.trace_fingerprint(p.video_path); d["prompts"] = [a.prompt for a in p.anchors]
os.makedirs(os.path.join(D, "items"), exist_ok=True)
json.dump(d, open(os.path.join(D, ir.TRACE_FILE), "w", encoding="utf-8"), ensure_ascii=False)
trace = ir.build_trace(p, None, None)       # should hit the cache
print("trace frames:", trace["frames"])
stats = ir.build_dataset(p, trace, seeds, progress=lambda a, b: None)
print("dataset:", stats)
reader.release()
