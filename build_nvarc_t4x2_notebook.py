#!/usr/bin/env python3
"""Build the attributed Kaggle NVARC T4x2 candidate."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
KAGGLE_DIR = HERE / "notebooks" / "kaggle_nvarc_t4x2"
UPSTREAM = KAGGLE_DIR / "upstream_source.ipynb"
NOTEBOOK = KAGGLE_DIR / "notebook.ipynb"
METADATA = KAGGLE_DIR / "kernel-metadata.json"
UPSTREAM_SHA256 = "452dbb1fc050bad9cbba38a6802231bff91cc2970899eabb8c74e5f34f322a6c"

# Kaggle account that owns the pushed kernels (see build_nvarc_notebook.py).
KAGGLE_OWNER = os.environ.get("ARC_KAGGLE_OWNER", "pramitdas")

ATTRIBUTION = """# ARC-AGI-2 — attributed NVARC T4x2 candidate

This private competition candidate reproduces the public Kaggle notebook
[`nihilisticneuralnet/baseline-nvarc-arc-25-winning-solution-for-t4x2`](https://www.kaggle.com/code/nihilisticneuralnet/baseline-nvarc-arc-25-winning-solution-for-t4x2),
which adapts the public NVARC solution lineage to two Tesla T4 GPUs.

Changes in this repository are limited to explicit attribution, a 6.5-hour
search deadline with a five-minute output buffer, and a worker count capped by
the GPUs Kaggle exposes. The proven 4-bit Qwen3-4B, eager-attention, 8-bit
optimizer, augmentation, LoRA, DFS, and two-guess selection settings are
preserved from the upstream notebook.
"""


def fail(message: str) -> None:
    raise SystemExit(f"build_nvarc_t4x2_notebook: {message}")


def validate_sources(cells: list[dict]) -> None:
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if source.startswith("!"):
            if "pip uninstall -y tensorflow" not in source and "UNSLOTH_DISABLE_STATISTICS" not in source:
                fail(f"unexpected shell cell {index}")
            continue
        if source.startswith("%%writefile "):
            source = source.split("\n", 1)[1]
        source = "\n".join(
            "" if line.lstrip().startswith(("!", "%%")) else line
            for line in source.splitlines()
        )
        ast.parse(source, filename=f"upstream-cell-{index}.py")


def main() -> None:
    if not UPSTREAM.is_file():
        fail(f"missing upstream source: {UPSTREAM}")
    import hashlib
    actual_hash = hashlib.sha256(UPSTREAM.read_bytes()).hexdigest()
    if actual_hash != UPSTREAM_SHA256:
        fail(f"upstream hash mismatch: {actual_hash}")

    notebook = json.loads(UPSTREAM.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    if len(cells) != 8:
        fail(f"expected 8 upstream cells, found {len(cells)}")

    metadata = dict(notebook.get("metadata", {}))
    metadata.pop("papermill", None)
    notebook["metadata"] = metadata
    for cell in cells:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        else:
            cell.pop("outputs", None)
            cell.pop("execution_count", None)

    time_cell = cells[0]
    time_source = "".join(time_cell["source"])
    old_budget = "global_end_time = time.time() + 12*3600 - 1200"
    if old_budget not in time_source:
        fail("upstream time budget did not match expected source")
    time_cell["source"] = time_source.replace(
        old_budget, "global_end_time = time.time() + 6.5 * 3600 - 300"
    )

    starter = "".join(cells[5]["source"])
    old_workers = """    for _ in range(2):
        queue.put(None)
    
    mp.spawn(local_worker, args=(queue, args.end_time), nprocs=2)"""
    new_workers = """    worker_count = max(1, min(2, torch.cuda.device_count()))
    print(f\"Using {worker_count} GPU worker(s)\", flush=True)
    for _ in range(worker_count):
        queue.put(None)
    
    mp.spawn(local_worker, args=(queue, args.end_time), nprocs=worker_count)"""
    if old_workers not in starter:
        fail("upstream T4x2 worker block did not match expected source")
    cells[5]["source"] = starter.replace(old_workers, new_workers)

    validate_sources(cells)

    # The public T4x2 notebook predates the current Kaggle model mount layout.
    # Keep the original source as provenance, but use the mounted model path
    # that is known to exist for the attached model source.
    solver_source = "".join(cells[4]["source"])
    old_model_path = "/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1"
    new_model_path = "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"
    if old_model_path not in solver_source:
        fail("upstream T4x2 model path did not match expected source")
    cells[4]["source"] = solver_source.replace(old_model_path, new_model_path)

    # A worker failure should still leave a schema-valid submission file.
    final_source = "".join(cells[7]["source"])
    old_store = 'decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)\n\ndecoder.load_decoded_results("/kaggle/inference_outputs")'
    new_store = 'decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)\n\nos.makedirs("/kaggle/inference_outputs", exist_ok=True)\ndecoder.load_decoded_results("/kaggle/inference_outputs")'
    if old_store not in final_source:
        fail("upstream final inference-output block did not match expected source")
    cells[7]["source"] = final_source.replace(old_store, new_store)

    validate_sources(cells)
    notebook["cells"].insert(
        0,
        {"cell_type": "markdown", "metadata": {}, "source": ATTRIBUTION},
    )
    nbformat.validate(nbformat.from_dict(notebook))
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    kernel_metadata = {
        "id": f"{KAGGLE_OWNER}/arc-2026-agi2-nvarc-t4x2-v1",
        "title": "arc-2026-agi2-nvarc-t4x2-v1",
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "false",
        "machine_shape": "NvidiaTeslaT4",
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868",
        "dataset_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-2"],
        "kernel_sources": ["sorokin/pip-install-unsloth-flash-patch"],
        "model_sources": [
            "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"
        ],
    }
    METADATA.write_text(json.dumps(kernel_metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {NOTEBOOK}")
    print(f"wrote {METADATA}")
    print(f"kernel owner: {KAGGLE_OWNER}")


if __name__ == "__main__":
    main()
