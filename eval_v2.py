import json, sys
sys.path.insert(0, "src")
from solver_v2 import solve_all_v2
from solver import score_predictions

for name, ch_path, sol_path in [
    ("eval-120", "data/arc-agi_evaluation_challenges.json", "data/arc-agi_evaluation_solutions.json"),
]:
    ch=json.load(open(ch_path)); sol=json.load(open(sol_path))
    preds, rules=solve_all_v2(ch)
    c,t,s=score_predictions(ch,sol,preds)
    print(f"{name}: {c}/{t}={s*100:.2f}%")
    from collections import Counter
    print(Counter(rules.values()).most_common(10))

# validate submission format on test set (240)
test=json.load(open("data/arc-agi_test_challenges.json"))
preds,_=solve_all_v2(test)
# strict checks per competition guidelines
assert len(preds)==len(test), "missing tasks"
for tid,task in test.items():
    assert tid in preds, tid
    assert len(preds[tid])==len(task["test"]), tid
    for p in preds[tid]:
        assert "attempt_1" in p and "attempt_2" in p, tid
        # grids must be list of lists of ints 0-9
        for k in ("attempt_1","attempt_2"):
            g=p[k]
            assert isinstance(g,list) and len(g)>0 and isinstance(g[0],list), tid
            for row in g:
                for v in row:
                    assert isinstance(v,int) and 0<=v<=9, tid
print("submission format OK for 240 test tasks")
json.dump(preds, open("output/submission_v2.json","w"))
print("wrote output/submission_v2.json")
