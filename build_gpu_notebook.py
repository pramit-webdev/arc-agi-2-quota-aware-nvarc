#!/usr/bin/env python3
"""Build GPU notebook: heuristics + SOAR LLM program synthesis (offline)."""
import json, os

code = r'''
import json, os, time, re, traceback
import numpy as np
print("ARC-AGI-2 GPU solver: heuristics + SOAR program synthesis (offline)")

CANDS=[
 "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json",
 "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json",
 "data/arc-agi_test_challenges.json",
]
IN_PATH=next((p for p in CANDS if os.path.exists(p)),None)
if IN_PATH is None and os.path.isdir("/kaggle/input"):
    for r,_,fs in os.walk("/kaggle/input"):
        for f in fs:
            if "test" in f and f.endswith(".json"):
                IN_PATH=os.path.join(r,f); break
print("INPUT:",IN_PATH)
tasks=json.load(open(IN_PATH))
print("tasks:",len(tasks))

def to_arr(g): return np.array(g,dtype=int)
def to_list(a): return np.array(a,dtype=int).tolist()

# ---- fast heuristics (same as CPU) ----
def op_id(a): return a.copy()
def op_r90(a): return np.rot90(a,1)
def op_r180(a): return np.rot90(a,2)
def op_r270(a): return np.rot90(a,3)
def op_fh(a): return np.fliplr(a)
def op_fv(a): return np.flipud(a)
def op_tr(a): return a.T.copy()
BASE=[op_id,op_r90,op_r180,op_r270,op_fh,op_fv,op_tr]
def recolor(a,mp):
    b=a.copy()
    for k,v in mp.items(): b[a==k]=v
    return b
def learn_map(ins,outs):
    mp={}
    for i,o in zip(ins,outs):
        if i.shape!=o.shape: return None
        for ci,co in zip(i.ravel(),o.ravel()):
            ci,co=int(ci),int(co)
            if ci in mp and mp[ci]!=co: return None
            mp[ci]=co
    return mp

def heuristic_solve(task):
    trains=[(to_arr(p["input"]),to_arr(p["output"])) for p in task["train"]]
    for op in BASE:
        try:
            if all(np.array_equal(op(i),o) for i,o in trains): return op, True
        except Exception: pass
    for op in BASE:
        try:
            mp=learn_map([op(i) for i,_ in trains],[o for _,o in trains])
            if mp is not None and all(np.array_equal(recolor(op(i),mp),o) for i,o in trains):
                return (lambda a,_o=op,_m=mp: recolor(_o(a),_m)), True
        except Exception: pass
    return op_id, False

# ---- LLM setup (defensive: fallback to heuristics if model missing) ----
USE_LLM=True
MODEL_DIRS=[]
for r,_,fs in os.walk("/kaggle/input" if os.path.isdir("/kaggle/input") else "."):
    # look for config.json (HF model)
    if "config.json" in fs and ("soar" in r.lower() or "qwen" in r.lower()):
        MODEL_DIRS.append(r)
print("candidate model dirs:",MODEL_DIRS[:5])
llm_pipe=None
if USE_LLM and MODEL_DIRS:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        md=sorted(MODEL_DIRS, key=lambda x: len(x))[0]
        print("loading",md,"cuda:",torch.cuda.is_available())
        tok=AutoTokenizer.from_pretrained(md, trust_remote_code=True)
        # 4-bit if available to fit T4
        try:
            from transformers import BitsAndBytesConfig
            q=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            model=AutoModelForCausalLM.from_pretrained(md, quantization_config=q, device_map="auto", trust_remote_code=True)
        except Exception as e:
            print("4bit failed, fp16:",e)
            model=AutoModelForCausalLM.from_pretrained(md, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        llm_pipe=pipeline("text-generation", model=model, tokenizer=tok, max_new_tokens=512, do_sample=True, temperature=0.8, top_p=0.95)
        print("LLM ready")
    except Exception as e:
        print("LLM load failed, heuristics only:",e)
        traceback.print_exc()
        llm_pipe=None
else:
    print("no model dir found -> heuristics only (attach pourceljulien/soar-qwen-7b as input for full power)")

def grid_to_py(g):
    return "["+",".join("["+",".join(str(int(v)) for v in row)+"]" for row in g)+"]"

PROMPT_TMPL="""You are an ARC-AGI code synthesizer. Write a Python function `solve(grid)` that maps input grid to output grid.
Grids are lists of lists of ints 0-9.
Examples:
{examples}
Test input:
input = {test_in}
Write ONLY valid Python code:
def solve(grid):
    ...
Code must use only numpy (as np) and stdlib, no I/O. Keep it general (no hard-coded test output).
"""

def run_code(code_str, grid):
    # restricted exec, 2s timeout via signal? use simple exec (Kaggle CPU ok)
    ns={"np":np}
    exec(code_str, ns)
    fn=ns.get("solve") or ns.get("transform")
    assert callable(fn), "no solve()"
    out=fn([row[:] for row in grid])
    return to_arr(out).tolist()

def llm_programs(task, n=8):
    trains=task["train"]; test_in=trains[0]["input"]  # placeholder, replaced per test below
    ex=[]
    for p in trains:
        ex.append(f"input = {grid_to_py(p['input'])}\noutput = {grid_to_py(p['output'])}")
    progs=[]
    for _ in range(n):
        prompt=PROMPT_TMPL.format(examples="\n".join(ex), test_in=grid_to_py(task["test"][0]["input"]))
        try:
            out=llm_pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.85, top_p=0.95, return_full_text=False)[0]["generated_text"]
            # extract code block
            m=re.findall(r"```python(.*?)```", out, re.S)
            code_txt=(m[0] if m else out)
            if "def solve" not in code_txt and "def transform" not in code_txt:
                continue
            progs.append(code_txt)
        except Exception:
            continue
    return progs

def score_prog(code_txt, trains):
    try:
        tot=0; ok=0
        for p in trains:
            pred=run_code(code_txt, p["input"])
            if np.array_equal(to_arr(pred), to_arr(p["output"])): ok+=1
            tot+=1
        return ok/tot
    except Exception:
        return -1

t0=time.time()
TIME_LIMIT=int(os.environ.get("SOLVE_TIME_LIMIT", 11*3600))
PER_TASK_LLM=6
sub={}
n_llm=0; n_heur=0
for idx,(tid,task) in enumerate(tasks.items()):
    if time.time()-t0>TIME_LIMIT-300:
        print("time guard: fallback remaining to heuristics")
        for rtid,rtask in list(tasks.items())[idx:]:
            fn,_=heuristic_solve(rtask)
            sub[rtid]=[{"attempt_1":to_list(fn(to_arr(t["input"]))),"attempt_2":to_list(to_arr(t["input"]))} for t in rtask["test"]]
        break
    fn, solved = heuristic_solve(task)
    if solved:
        n_heur+=1
        sub[tid]=[{"attempt_1":to_list(fn(to_arr(t["input"]))),"attempt_2":to_list(to_arr(t["input"]))} for t in task["test"]]
        continue
    # need LLM
    if llm_pipe is None:
        sub[tid]=[{"attempt_1":to_list(to_arr(t["input"])),"attempt_2":to_list(op_r90(to_arr(t["input"])))} for t in task["test"]]
        continue
    n_llm+=1
    try:
        progs=llm_programs(task, n=PER_TASK_LLM)
        scored=[]
        for pr in progs:
            s=score_prog(pr, task["train"])
            scored.append((s,pr))
        scored.sort(reverse=True, key=lambda x: x[0])
        # pick best that passes all train, else best-effort
        best=None; second=None
        for s,pr in scored:
            if s==1.0:
                best=pr; break
        if best is None and scored and scored[0][0]>=0:
            best=scored[0][1]
        # build predictions per test input
        preds=[]
        for t in task["test"]:
            tin=t["input"]
            p1=p2=None
            if best is not None:
                try: p1=run_code(best, tin)
                except Exception: p1=None
            # second: next best different output or input copy
            for s,pr in scored:
                if pr==best: continue
                try:
                    cand=run_code(pr, tin)
                    if cand!=p1:
                        p2=cand; break
                except Exception: continue
            if p1 is None: p1=tin
            if p2 is None: p2=to_list(op_r90(to_arr(tin)))
            preds.append({"attempt_1":p1,"attempt_2":p2})
        sub[tid]=preds
    except Exception as e:
        print(tid,"llm fail",e)
        sub[tid]=[{"attempt_1":t["input"],"attempt_2":to_list(op_r90(to_arr(t["input"])))} for t in task["test"]]
    if (idx+1)%20==0:
        print(f"{idx+1}/{len(tasks)} heur={n_heur} llm={n_llm} {time.time()-t0:.0f}s")
print(f"done heur={n_heur} llm_tasks={n_llm} total {time.time()-t0:.1f}s")
# validate + write
assert len(sub)==len(tasks)
for tid,task in tasks.items():
    assert len(sub[tid])==len(task["test"])
    for p in sub[tid]:
        assert "attempt_1" in p and "attempt_2" in p
print("VALIDATION PASSED")
for pth in ["/kaggle/working/submission.json","submission.json"]:
    try: json.dump(sub,open(pth,"w")); print("wrote",pth)
    except Exception as e: print("skip",pth,e)
'''

nb={"cells":[
 {"cell_type":"markdown","metadata":{},"source":["# ARC-AGI-2 GPU — heuristics + SOAR program synthesis (prize-eligible)\n","Offline. Attach model `pourceljulien/soar-qwen-7b` as input + competition data. Falls back to heuristics if model missing.\n","License: MIT."]},
 {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":code.splitlines(True)},
],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.10"}},"nbformat":4,"nbformat_minor":5}
os.makedirs("notebooks/kaggle_gpu",exist_ok=True)
json.dump(nb,open("notebooks/kaggle_gpu/notebook.ipynb","w"),indent=1)
meta={"id":"pramitdas/arc-2026-agi2-gpu-soar","title":"arc-2026-agi2-gpu-soar","code_file":"notebook.ipynb","language":"python","kernel_type":"notebook","is_private":"true","enable_gpu":"true","enable_tpu":"false","enable_internet":"false","dataset_sources":[],"competition_sources":["arc-prize-2026-arc-agi-2"],"kernel_sources":[],"model_sources":["pourceljulien/soar-qwen-7b/transformers/default/1"]}
json.dump(meta,open("notebooks/kaggle_gpu/kernel-metadata.json","w"),indent=1)
print("wrote gpu notebook + metadata")
