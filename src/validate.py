"""Validator for submission.json per ARC Prize 2026 guidelines."""
import json, sys
def validate(sub_path, chal_path):
    sub=json.load(open(sub_path))
    chal=json.load(open(chal_path))
    assert isinstance(sub, dict), "root must be dict"
    assert set(sub.keys())==set(chal.keys()), f"task id mismatch: {len(sub)} vs {len(chal)}"
    for tid, task in chal.items():
        preds=sub[tid]
        assert isinstance(preds, list) and len(preds)==len(task["test"]), f"{tid} num preds"
        for p in preds:
            assert "attempt_1" in p and "attempt_2" in p, tid
            for k in ("attempt_1","attempt_2"):
                g=p[k]
                assert isinstance(g,list) and g and isinstance(g[0],list), tid
                w=len(g[0])
                for row in g:
                    assert len(row)==w and all(isinstance(v,int) and 0<=v<=9 for v in row), tid
    print(f"VALID: {len(sub)} tasks, all checks passed: {sub_path}")
if __name__=="__main__":
    validate(sys.argv[1] if len(sys.argv)>1 else "output/submission_v2.json",
             sys.argv[2] if len(sys.argv)>2 else "data/arc-agi_test_challenges.json")
