"""Evaluate solver on public eval set (120 tasks, low RAM)."""
import json, sys
sys.path.insert(0, "src")
from solver import solve_all, score_predictions, to_arr

with open("data/arc-agi_evaluation_challenges.json") as f:
    eval_ch = json.load(f)
with open("data/arc-agi_evaluation_solutions.json") as f:
    eval_sol = json.load(f)

print(f"Eval tasks: {len(eval_ch)}")
preds, rules = solve_all(eval_ch)
c, t, s = score_predictions(eval_ch, eval_sol, preds)
print(f"Score: {c}/{t} = {s*100:.2f}%")

from collections import Counter
print(Counter(rules.values()).most_common(15))

# show first 5 failures for analysis
shown = 0
for tid in list(eval_ch.keys())[:120]:
    sols = eval_sol[tid]
    for i, sol in enumerate(sols):
        import numpy as np
        ok = np.array_equal(to_arr(preds[tid][i]["attempt_1"]), to_arr(sol)) or \
             np.array_equal(to_arr(preds[tid][i]["attempt_2"]), to_arr(sol))
        if not ok and shown < 5:
            print(f"FAIL {tid} rule={rules[tid]} in={to_arr(eval_ch[tid]['test'][i]['input']).shape} sol={to_arr(sol).shape}")
            shown += 1
