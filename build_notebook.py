#!/usr/bin/env python3
"""Build self-contained Kaggle CPU notebook (offline, no internet, numpy-only)."""
import json

# Notebook code as a single .py-style cell (kept readable, low-RAM)
code = r'''
import json, os, time, itertools
import numpy as np

print("ARC Prize 2026 ARC-AGI-2 - CPU baseline (numpy-only, offline)")

# ---------- input discovery (Kaggle competition mount variants + local fallback) ----------
CANDIDATES = [
    "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json",
    "/kaggle/input/arc-prize-2026-arc-agi-2/test_challenges.json",
    "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json",
    "data/arc-agi_test_challenges.json",
    "../input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json",
]
IN_PATH = next((p for p in CANDIDATES if os.path.exists(p)), None)
# last resort: scan /kaggle/input for any *test*challenges*.json
if IN_PATH is None and os.path.isdir("/kaggle/input"):
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            if "test" in f and f.endswith(".json"):
                IN_PATH = os.path.join(root, f)
                break
assert IN_PATH is not None, f"test challenges not found, tried {CANDIDATES}"
print("INPUT:", IN_PATH)
tasks = json.load(open(IN_PATH))
print(f"tasks: {len(tasks)}")

def to_arr(g): return np.array(g, dtype=int)
def to_list(a): return np.array(a, dtype=int).tolist()
def op_id(a): return a.copy()
def op_r90(a): return np.rot90(a,1)
def op_r180(a): return np.rot90(a,2)
def op_r270(a): return np.rot90(a,3)
def op_fh(a): return np.fliplr(a)
def op_fv(a): return np.flipud(a)
def op_tr(a): return a.T.copy()
BASE=[("id",op_id),("r90",op_r90),("r180",op_r180),("r270",op_r270),("fh",op_fh),("fv",op_fv),("tr",op_tr)]
def crop_bg(a,bg=0):
    m=a!=bg
    if not m.any(): return a.copy()
    ys,xs=np.where(m)
    return a[ys.min():ys.max()+1,xs.min():xs.max()+1].copy()
def left_half(a): return a[:,:max(1,a.shape[1]//2)].copy()
def right_half(a):
    w=a.shape[1]; return a[:,(w+1)//2:].copy() if w>1 else a.copy()
def top_half(a): return a[:max(1,a.shape[0]//2),:].copy()
def bottom_half(a):
    h=a.shape[0]; return a[(h+1)//2:,:].copy() if h>1 else a.copy()
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

def cell_acc(a,b):
    if a.shape!=b.shape: return -1.0
    return float((a==b).mean())

def solve_one(task):
    trains=[(to_arr(p["input"]),to_arr(p["output"])) for p in task["train"]]
    tests=[to_arr(p["input"]) for p in task["test"]]
    cands=[]
    for n,op in BASE:
        try:
            if all(np.array_equal(op(i),o) for i,o in trains): cands.append((n,op,2))
        except Exception: pass
    for n,op in BASE:
        try:
            mp=learn_map([op(i) for i,_ in trains],[o for _,o in trains])
            if mp is not None and all(np.array_equal(recolor(op(i),mp),o) for i,o in trains):
                cands.append((f"{n}+map",lambda a,_o=op,_m=mp: recolor(_o(a),_m),2))
        except Exception: pass
    for sn,sop in [("crop",crop_bg),("left",left_half),("right",right_half),("top",top_half),("bottom",bottom_half)]:
        try:
            if all(np.array_equal(sop(i),o) for i,o in trains): cands.append((sn,sop,1))
        except Exception: pass
        for bn,bop in BASE:
            try:
                if all(np.array_equal(bop(sop(i)),o) for i,o in trains):
                    cands.append((f"{sn}+{bn}",lambda a,_s=sop,_b=bop:_b(_s(a)),1))
            except Exception: pass
    # if exact rule found -> use it; else pick best by train cell-accuracy among simple guesses
    if cands:
        # prefer higher priority then shorter name
        cands.sort(key=lambda x:(-x[2],len(x[0])))
        best=cands[0]
        second=cands[1] if len(cands)>1 else ("id",op_id,0)
    else:
        # search best-effort: score identity/rot/flip + recolor-fit by accuracy
        pool=[("id",op_id)]
        for n,op in BASE[1:]: pool.append((n,op))
        best_score=-2; best=("id",op_id,0); second=("r90",op_r90,0)
        scored=[]
        for n,op in pool:
            try:
                s=sum(cell_acc(op(i),o) for i,o in trains)/len(trains)
            except Exception: s=-2
            scored.append((s,n,op))
        scored.sort(reverse=True)
        best=(scored[0][1],scored[0][2],0)
        second=(scored[1][1],scored[1][2],0) if len(scored)>1 else best
    out=[]
    for t in tests:
        try: p1=to_list(best[1](t))
        except Exception: p1=to_list(t)
        try: p2=to_list(second[1](t))
        except Exception: p2=to_list(t)
        out.append({"attempt_1":p1,"attempt_2":p2})
    return out

t0=time.time()
sub={}
for i,(tid,task) in enumerate(tasks.items()):
    sub[tid]=solve_one(task)
    if (i+1)%50==0: print(f"  {i+1}/{len(tasks)} done, {time.time()-t0:.1f}s")
print(f"solved {len(sub)} tasks in {time.time()-t0:.1f}s")

# ---------- strict validation per competition guidelines ----------
assert len(sub)==len(tasks)
for tid,task in tasks.items():
    assert tid in sub and len(sub[tid])==len(task["test"])
    for p in sub[tid]:
        assert "attempt_1" in p and "attempt_2" in p
        for k in ("attempt_1","attempt_2"):
            g=p[k]
            assert isinstance(g,list) and g and isinstance(g[0],list)
            w=len(g[0])
            for row in g:
                assert len(row)==w and all(isinstance(v,int) and 0<=v<=9 for v in row)
print("VALIDATION PASSED")

for p in ["/kaggle/working/submission.json","submission.json"]:
    try:
        json.dump(sub,open(p,"w"))
        print("wrote",p)
    except Exception as e:
        print("skip",p,e)
print("DONE")
'''

nb = {
    "cells": [
        {"cell_type": "markdown", "metadata": {},
         "source": ["# ARC Prize 2026 ARC-AGI-2 — CPU baseline (prize-eligible)\n",
                    "Offline, numpy-only, ≤12h CPU, outputs `submission.json` with 2 attempts per test.\n",
                    "License: MIT (required for prize eligibility — open source)."]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)},
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

import os
os.makedirs("notebooks/kaggle_cpu", exist_ok=True)
json.dump(nb, open("notebooks/kaggle_cpu/notebook.ipynb", "w"), indent=1)
print("wrote notebooks/kaggle_cpu/notebook.ipynb")

meta = {
    "id": "pramitdas/arc-2026-agi2-cpu-baseline",
    "title": "arc-2026-agi2-cpu-baseline",
    "code_file": "notebook.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "false",
    "enable_tpu": "false",
    "enable_internet": "false",
    "dataset_sources": ["arc-prize-2026-arc-agi-2"],
    "competition_sources": ["arc-prize-2026-arc-agi-2"],
    "kernel_sources": []
}
json.dump(meta, open("notebooks/kaggle_cpu/kernel-metadata.json", "w"), indent=1)
print("wrote kernel-metadata.json")
