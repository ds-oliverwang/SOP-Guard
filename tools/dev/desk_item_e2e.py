"""After training: install the model into the project, generate item steps, run the real monitor over the whole desk
video with --state-log, and summarise per round which items were seen and what the checklist decided."""
import os, sys, json, shutil, subprocess, csv
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import training, item_recognizer as ir
from project_store import load_project
D = os.path.join(ROOT, "sop_projects", "桌面狀態測試")
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "cache", "item_model", "best.pt")
os.makedirs(os.path.join(D, ir.MODEL_DIR), exist_ok=True)
shutil.copyfile(src, os.path.join(D, ir.MODEL_FILE))
p = load_project(D); seeds = ir.ItemSeeds(D)
# 物件版：state 範例留著也沒關係，但步驟改成物品
names = training.apply_item_steps(p, seeds.order)
p = load_project(D)
print("steps:", [(s.type, s.label) for s in p.steps], "end:", p.cycle_end.type, p.cycle_end.object)
PY = sys.executable
log = os.path.join(ROOT, "cache", "desk_item_state_log.csv")
r = subprocess.run([PY, "core_detector.py", "--project", D, "--source", p.video_path, "--state-log", log, "--save-preview", os.path.join(ROOT, "cache", "preview_items")],
                   capture_output=True, text=True, encoding="utf-8", errors="replace", env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
for line in r.stdout.splitlines():
    if line.startswith("[t=") or line.startswith("cycles="):
        print(line)
# per-round item visibility from the log (judged rows with items)
ROUNDS = [(830, 1185), (1900, 2165), (2710, 3045), (3550, 3980), (4520, 4950)]
rows = list(csv.DictReader(open(log, encoding="utf-8-sig")))
for i, (f0, f1) in enumerate(ROUNDS, 1):
    seen = {}; judged = 0
    for row in rows:
        f = int(row["frame"])
        if f0 <= f <= f1 and row["why"] == "judged":
            judged += 1
            for it in (row.get("items") or "").split("|"):
                if it:
                    seen[it] = seen.get(it, 0) + 1
    print(f"round {i} f{f0}-{f1}: judged {judged} frames, seen " + ", ".join(f"{k}×{v}" for k, v in sorted(seen.items())))
