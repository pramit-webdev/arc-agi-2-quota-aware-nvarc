#!/usr/bin/env python3
import json, os
code = r'''
import os, time, traceback
print("PROBE start", flush=True)
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), flush=True)
MODEL_DIRS=[]
for r,_,fs in os.walk("/kaggle/input"):
    if "config.json" in fs and ("soar" in r.lower() or "qwen" in r.lower()):
        if any(f.endswith(".safetensors") or f.endswith(".bin") for f in fs):
            MODEL_DIRS.append(r)
print("models:", MODEL_DIRS, flush=True)
assert MODEL_DIRS, "no model found"
from transformers import AutoTokenizer, AutoModelForCausalLM
md=sorted(MODEL_DIRS,key=len)[0]
print("loading", md, flush=True)
t0=time.time()
tok=AutoTokenizer.from_pretrained(md, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(md, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True)
model.eval()
print(f"loaded in {time.time()-t0:.1f}s device={next(model.parameters()).device}", flush=True)
ids=tok("def solve(grid):\n return grid", return_tensors="pt")
ids={k:v.to(model.device) for k,v in ids.items()}
t1=time.time()
with torch.no_grad():
    out=model.generate(**ids, max_new_tokens=32, do_sample=False)
print(f"gen 32 tok in {time.time()-t1:.1f}s", flush=True)
print(tok.decode(out[0], skip_special_tokens=True)[-200:], flush=True)
print("PROBE OK", flush=True)
'''
nb={"cells":[
 {"cell_type":"markdown","metadata":{},"source":["# Probe T4 + SOAR (2 min)\n"]},
 {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":code.splitlines(True)}],
 "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.10"}},
 "nbformat":4,"nbformat_minor":5}
os.makedirs("notebooks/kaggle_probe",exist_ok=True)
json.dump(nb,open("notebooks/kaggle_probe/notebook.ipynb","w"),indent=1)
meta={"id":"pramitdas/arc-probe-t4-soar","title":"arc-probe-t4-soar","code_file":"notebook.ipynb","language":"python","kernel_type":"notebook","is_private":"true","enable_gpu":"true","enable_tpu":"false","enable_internet":"false","machine_shape":"NvidiaTeslaT4","dataset_sources":[],"competition_sources":["arc-prize-2026-arc-agi-2"],"kernel_sources":[],"model_sources":["pourceljulien/soar-qwen-7b/transformers/default/1"]}
json.dump(meta,open("notebooks/kaggle_probe/kernel-metadata.json","w"),indent=1)
print("probe built")
