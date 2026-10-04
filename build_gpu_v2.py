#!/usr/bin/env python3
"""Build GPU v2: T4 + real SOAR sampling/refinement, offline, time-bounded."""
import json, os
code = r'''
import json, os, time, re, traceback
import numpy as np
print("ARC-AGI-2 GPU v2 (T4, SOAR sample+refine, offline)")

CANDS=["/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json",
 "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json",
 "data/arc-agi_test_challenges.json"]
IN_PATH=next((p for p in CANDS if os.path.exists(p)),None)
if IN_PATH is None and os.path.isdir("/kaggle/input"):
    for r,_,fs in os.walk("/kaggle/input"):
        for f in fs:
            if "test" in f and f.endswith(".json"):
                IN_PATH=os.path.join(r,f); break
print("INPUT:",IN_PATH)
tasks=json.load(open(IN_PATH))
print("tasks:",len(tasks))
import torch
print("torch",torch.__version__,"cuda",torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

def to_arr(g): return np.array(g,dtype=int)
def to_list(a): return np.array(a,dtype=int).tolist()
def op_id(a): return a.copy()
def op_r90(a): return np.rot90(a,1)
def op_r180(a): return np.rot90(a,2)
def op_r270(a): return np.rot90(a,3)
def op_fh(a): return np.fliplr(a)
def op_fv(a): return np.flipud(a)
def op_tr(a): return a.T.copy()
BASE_FNS=[op_id,op_r90,op_r180,op_r270,op_fh,op_fv,op_tr]
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

def heuristic_fn(task):
    trains=[(to_arr(p["input"]),to_arr(p["output"])) for p in task["train"]]
    for op in BASE_FNS:
        try:
            if all(np.array_equal(op(i),o) for i,o in trains): return op,True
        except Exception: pass
    for op in BASE_FNS:
        try:
            mp=learn_map([op(i) for i,_ in trains],[o for _,o in trains])
            if mp is not None and all(np.array_equal(recolor(op(i),mp),o) for i,o in trains):
                return (lambda a,_o=op,_m=mp: recolor(_o(a),_m)),True
        except Exception: pass
    return op_id,False

# ---- model ----
MODEL_DIRS=[]
if os.path.isdir("/kaggle/input"):
    for r,_,fs in os.walk("/kaggle/input"):
        if "config.json" in fs and ("soar" in r.lower() or "qwen" in r.lower()):
            # must contain model weights (safetensors/bin)
            has_w=any(f.endswith(".safetensors") or f.endswith(".bin") for f in fs)
            if has_w: MODEL_DIRS.append(r)
print("model dirs:",MODEL_DIRS)
tok=None; model=None
fail_logged=False
if MODEL_DIRS:
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        md=sorted(MODEL_DIRS,key=len)[0]
        print("loading",md)
        tok=AutoTokenizer.from_pretrained(md,trust_remote_code=True)
        if tok.pad_token is None: tok.pad_token=tok.eos_token
        model=AutoModelForCausalLM.from_pretrained(md,torch_dtype=torch.float16,device_map="auto",trust_remote_code=True,low_cpu_mem_usage=True)
        model.eval()
        print("model loaded, device:",next(model.parameters()).device)
        # smoke test (1 generation, tiny) to fail fast with visible error
        with torch.no_grad():
            ids=tok("def solve(grid):\n return grid",return_tensors="pt")
            ids={k:v.to(model.device) for k,v in ids.items()}
            out=model.generate(**ids,max_new_tokens=16,do_sample=False)
            print("smoke OK:",tok.decode(out[0],skip_special_tokens=True)[-80:])
    except Exception as e:
        print("MODEL LOAD/SMOKE FAILED:",e); traceback.print_exc()
        tok=None; model=None
else:
    print("no model -> heuristics only")

def grid_py(g):
    return "["+",".join("["+",".join(str(int(v)) for v in row)+"]" for row in g)+"]"

SAMPLE_TMPL="""Write Python `def solve(grid):` mapping ARC input grid to output grid. Grids are lists of lists ints 0-9. Use only numpy as np.
{examples}
Test input: input = {test_in}
Output ONLY code:
def solve(grid):
"""
REFINE_TMPL="""Fix Python `def solve(grid):` for ARC. It failed training examples.
{examples}
Failed code:
{bad_code}
Error/mismatch: {feedback}
Write corrected `def solve(grid):` only, general (no hard-coding), numpy as np allowed.
def solve(grid):
"""

def generate(prompt, max_new=384, temp=0.85):
    ids=tok(prompt,return_tensors="pt",truncation=True,max_length=3000)
    ids={k:v.to(model.device) for k,v in ids.items()}
    with torch.no_grad():
        out=model.generate(**ids,max_new_tokens=max_new,do_sample=(temp>0),temperature=temp,top_p=0.95,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    txt=tok.decode(out[0][ids["input_ids"].shape[1]:],skip_special_tokens=True)
    return txt

def extract_fn(txt):
    m=re.findall(r"```python(.*?)```",txt,re.S)
    c=(m[0] if m else txt)
    # keep from def solve
    i=c.find("def solve")
    j=c.find("def transform")
    k=i if i>=0 else j
    if k>=0: c=c[k:]
    return c

def run_code(code_txt, grid):
    ns={"np":np}
    exec(code_txt,ns)
    fn=ns.get("solve") or ns.get("transform")
    assert callable(fn),"no solve()"
    import signal
    out=fn([row[:] for row in grid])
    return to_list(to_arr(out))

def train_score(code_txt, trains):
    try:
        ok=0
        for p in trains:
            if np.array_equal(to_arr(run_code(code_txt,p["input"])),to_arr(p["output"])): ok+=1
        return ok/len(trains)
    except Exception:
        return -1

t0=time.time()
TIME_LIMIT=int(os.environ.get("SOLVE_TIME_LIMIT",11*3600))
# budget: easy tasks skip LLM; hard tasks get samples+refines; total must fit 11h
N_SAMPLE=int(os.environ.get("N_SAMPLE","24"))
N_REFINE=int(os.environ.get("N_REFINE","12"))
print(f"budget sample={N_SAMPLE} refine={N_REFINE}")
sub={}; nH=0; nL=0; nVerified=0
for idx,(tid,task) in enumerate(tasks.items()):
    if time.time()-t0>TIME_LIMIT-600:
        print("TIME GUARD fallback rest")
        for rtid,rtask in list(tasks.items())[idx:]:
            fn,_=heuristic_fn(rtask)
            sub[rtid]=[{"attempt_1":to_list(fn(to_arr(t["input"]))),"attempt_2":to_list(to_arr(t["input"]))} for t in rtask["test"]]
        break
    fn,solved=heuristic_fn(task)
    if solved:
        nH+=1
        sub[tid]=[{"attempt_1":to_list(fn(to_arr(t["input"]))),"attempt_2":to_list(to_arr(t["input"]))} for t in task["test"]]
        continue
    if model is None or tok is None:
        sub[tid]=[{"attempt_1":t["input"],"attempt_2":to_list(op_r90(to_arr(t["input"])))} for t in task["test"]]
        continue
    nL+=1
    try:
        ex="\n".join(f"input = {grid_py(p['input'])}\noutput = {grid_py(p['output'])}" for p in task["train"])
        cands=[]
        # sample
        for _ in range(N_SAMPLE):
            if time.time()-t0>TIME_LIMIT-600: break
            try:
                txt=generate(SAMPLE_TMPL.format(examples=ex,test_in=grid_py(task["test"][0]["input"])),max_new=384,temp=0.9)
                c=extract_fn(txt)
                if "def " not in c: continue
                s=train_score(c,task["train"])
                if s>=0: cands.append((s,c))
                if s==1.0: break
            except Exception as e:
                global fail_logged
                if not fail_logged:
                    print("gen fail eg:",e); traceback.print_exc(); fail_logged=True
                continue
        cands.sort(reverse=True,key=lambda x:x[0])
        # refine top few non-perfect
        refined=[]
        for s,c in cands[:6]:
            if s==1.0: refined.append((s,c)); continue
            if len(refined)+len(cands)>=N_SAMPLE+N_REFINE: break
            try:
                fb=f"train score {s:.2f}"
                txt=generate(REFINE_TMPL.format(examples=ex,bad_code=c[:2000],feedback=fb),max_new=384,temp=0.7)
                c2=extract_fn(txt)
                if "def " not in c2: continue
                s2=train_score(c2,task["train"])
                if s2>=0: refined.append((s2,c2))
            except Exception: continue
        pool=sorted(cands+refined,reverse=True,key=lambda x:x[0])
        if pool and pool[0][0]==1.0: nVerified+=1
        # predictions per test
        preds=[]
        for t in task["test"]:
            tin=t["input"]; p1=None; p2=None
            for s,c in pool:
                try:
                    o=run_code(c,tin)
                    if p1 is None: p1=o
                    elif o!=p1:
                        p2=o; break
                except Exception: continue
            if p1 is None: p1=tin
            if p2 is None:
                try: p2=to_list(op_r90(to_arr(tin)))
                except Exception: p2=tin
            preds.append({"attempt_1":p1,"attempt_2":p2})
        sub[tid]=preds
    except Exception as e:
        print(tid,"task fail",e)
        sub[tid]=[{"attempt_1":t["input"],"attempt_2":to_list(op_r90(to_arr(t["input"])))} for t in task["test"]]
    if (idx+1)%10==0 or (idx+1)==len(tasks):
        print(f"{idx+1}/{len(tasks)} H={nH} LLM={nL} verified={nVerified} {time.time()-t0:.0f}s",flush=True)
print(f"DONE H={nH} LLM={nL} verified={nVerified} {time.time()-t0:.1f}s")
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
 {"cell_type":"markdown","metadata":{},"source":["# ARC-AGI-2 GPU v2 — T4 SOAR sample+refine (prize-eligible)\n","Attach `pourceljulien/soar-qwen-7b` + competition data. Offline, no pip. Falls back safely.\n","License: MIT."]},
 {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":code.splitlines(True)}],
 "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.10"}},
 "nbformat":4,"nbformat_minor":5}
os.makedirs("notebooks/kaggle_gpu_v2",exist_ok=True)
json.dump(nb,open("notebooks/kaggle_gpu_v2/notebook.ipynb","w"),indent=1)
meta={"id":"pramitdas/arc-2026-agi2-gpu-v2","title":"arc-2026-agi2-gpu-v2","code_file":"notebook.ipynb","language":"python","kernel_type":"notebook","is_private":"true","enable_gpu":"true","enable_tpu":"false","enable_internet":"false","machine_shape":"NvidiaTeslaT4","dataset_sources":[],"competition_sources":["arc-prize-2026-arc-agi-2"],"kernel_sources":[],"model_sources":["pourceljulien/soar-qwen-7b/transformers/default/1"]}
json.dump(meta,open("notebooks/kaggle_gpu_v2/kernel-metadata.json","w"),indent=1)
print("wrote v2")
