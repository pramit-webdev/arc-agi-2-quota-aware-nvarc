#!/usr/bin/env python3
"""Offline evaluation harness for NVARC candidate pools.

Consumes the candidate-pool directories produced by the harvest notebook
(``notebooks/kaggle_nvarc_harvest``; see ``build_nvarc_notebook.py --harvest``)
and benchmarks retrieval/aggregation strategies against the public evaluation
labels.  Pure numpy so it runs on a low-RAM machine (no torch/transformers).

Pool files are ``bz2``-compressed pickles of a list of candidate dictionaries::

    {"beam_score": float, "score_aug": [float, ...], "solution": ndarray}

The filename is ``<tid>_<test_index>.<aug1>.<aug2>`` for the pooled run, or
``<tid>_<test_index>.<aug1>.<aug2>.out<N>`` for extra runs (as written by
``ArcDecoder.load_decoded_results``).  Candidates for the same task are grouped
exactly like the upstream decoder so the numbers here line up with the
competition notebook's own ``benchmark_selection_algos``.

Usage::

    python eval_nvarc_offline.py --store /tmp/harvest/inference_outputs \\
        --solutions data/arc-agi_evaluation_solutions.json
"""
from __future__ import annotations

import argparse
import bz2
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np


def to_code(grid):
    return tuple(tuple(int(v) for v in row) for row in np.asarray(grid).tolist())


def load_runs(store):
    """Return ``{run_name: {base_key: [candidate, ...]}}``."""
    runs = defaultdict(dict)
    for name in sorted(os.listdir(store)):
        path = os.path.join(store, name)
        if not os.path.isfile(path):
            continue
        base, dot, suffix = name.partition(".")
        if not dot:
            continue
        run_name = suffix.split(".out", 1)[0] if ".out" in suffix else ""
        try:
            with bz2.BZ2File(path) as fh:
                outputs = pickle.load(fh)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"skip {name}: {exc}", file=sys.stderr)
            continue
        run = runs[run_name]
        for sample in outputs:
            solution = sample.get("solution")
            if solution is None:
                continue
            run.setdefault(base, []).append(
                {
                    "code": to_code(solution),
                    "beam_score": float(sample.get("beam_score", 0.0)),
                    "score_aug": [float(s) for s in (sample.get("score_aug") or [])],
                }
            )
    return dict(runs)


def _group_candidates(pool):
    """Group candidates that propose the same output grid."""
    groups = {}
    for cand in pool:
        groups.setdefault(cand["code"], []).append(cand)
    return groups


def order_beams(pool):
    return [cand["code"] for cand in sorted(pool, key=lambda g: g["beam_score"])]


def order_full_probmul_3(pool, baseline=3.0):
    """Exact replication of arc_decoder.getter_full_probmul_3."""
    scored = []
    for code, guesses in _group_candidates(pool).items():
        inf_score = sum(baseline - g["beam_score"] for g in guesses)
        aug_score = float(
            np.mean([sum(baseline - s for s in g["score_aug"]) for g in guesses])
        )
        scored.append((inf_score + aug_score, code))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [code for _, code in scored]


def order_kgmon(pool):
    """Exact replication of arc_decoder.getter_kgmon."""
    scored = []
    for code, guesses in _group_candidates(pool).items():
        inf_score = len(guesses)
        aug_score = float(np.mean([np.mean(g["score_aug"]) for g in guesses]))
        scored.append((inf_score - aug_score, code))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [code for _, code in scored]


def order_uniform_vote(pool):
    scored = [
        (len(guesses), code) for code, guesses in _group_candidates(pool).items()
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [code for _, code in scored]


ORDERS = {
    "beam": order_beams,
    "probmul": order_full_probmul_3,
    "kgmon": order_kgmon,
    "uniform": order_uniform_vote,
}

STRATEGIES = [
    ("beam+beam", "beam", "beam"),
    ("probmul+probmul", "probmul", "probmul"),
    ("kgmon+kgmon", "kgmon", "kgmon"),
    ("probmul+kgmon", "probmul", "kgmon"),
    ("kgmon+probmul", "kgmon", "probmul"),
    ("probmul+uniform", "probmul", "uniform"),
    ("uniform+probmul", "uniform", "probmul"),
]

def two_guesses(pool, order1, order2):
    first_order = order1(pool)
    if not first_order:
        return None, None
    guess1 = first_order[0]
    for code in order2(pool):
        if code != guess1:
            return guess1, code
    return guess1, guess1


def load_labels(solutions_path):
    with open(solutions_path) as fh:
        solutions = json.load(fh)
    labels = {}
    for tid, items in solutions.items():
        for idx, grid in enumerate(items):
            labels[f"{tid}_{idx}"] = to_code(grid)
    return labels


def evaluate(runs, labels, ks=(1, 2, 3, 4, 6, 8)):
    for run_name, tasks in runs.items():
        labelled = {k: c for k, c in tasks.items() if k in labels}
        n_tasks = len(labelled)
        if not n_tasks:
            print(f"[{run_name or 'pooled'}] no labelled tasks found")
            continue
        candidates = sum(len(c) for c in labelled.values())
        print(
            f"\n== run '{run_name or 'pooled'}' ==  tasks={n_tasks}"
            f"  candidates={candidates}  avg={candidates / n_tasks:.1f}/task"
        )

        coverage = 0
        coverage_at = defaultdict(int)
        for key, cands in labelled.items():
            label = labels[key]
            if label not in {cand["code"] for cand in cands}:
                continue
            coverage += 1
            for order_name, order in ORDERS.items():
                ordered = order(cands)
                if label in ordered:
                    rank = ordered.index(label) + 1
                    for k in ks:
                        if rank <= k:
                            coverage_at[(order_name, k)] += 1
        print(
            f"pool coverage (label anywhere): {coverage}/{n_tasks}"
            f" = {100 * coverage / n_tasks:.2f}%"
        )
        for order_name in ORDERS:
            curve = "  ".join(
                f"k={k}:{100 * coverage_at[(order_name, k)] / n_tasks:5.1f}%"
                for k in ks
            )
            print(f"  best-order[{order_name:7s}] {curve}")

        print("  strategies (n_guess=2 competition metric):")
        for name, o1_name, o2_name in STRATEGIES:
            hits1 = hits2 = 0
            for key, cands in labelled.items():
                label = labels[key]
                g1, g2 = two_guesses(cands, ORDERS[o1_name], ORDERS[o2_name])
                if g1 is None:
                    continue
                if g1 == label:
                    hits1 += 1
                if g1 == label or g2 == label:
                    hits2 += 1
            print(
                f"    {name:18s} n1={hits1:3d} ({100 * hits1 / n_tasks:5.2f}%)"
                f"   n2={hits2:3d} ({100 * hits2 / n_tasks:5.2f}%)"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, help="pool directory to read")
    parser.add_argument(
        "--solutions",
        default="data/arc-agi_evaluation_solutions.json",
        help="label source (public evaluation solutions)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.store):
        parser.error(f"store directory not found: {args.store}")
    labels = load_labels(args.solutions)
    runs = load_runs(args.store)
    if not runs:
        parser.error(f"no pool files found in {args.store}")
    print(f"store: {args.store}")
    print(f"runs discovered: {sorted(runs)}  labels: {len(labels)}")
    evaluate(runs, labels)


if __name__ == "__main__":
    main()
