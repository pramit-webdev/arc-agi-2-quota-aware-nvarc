#!/usr/bin/env python3
"""Grid-search "picker" (selector) variants against harvested NVARC pools.

Reuses eval_nvarc_offline's pool/label loading.  For every candidate scorer it
prints n_guess=1 and n_guess=2 accuracy (the competition metric), sorted by
n_guess=2.  Higher = better.

A scorer ranks the candidate grids for one task; the top two become the two
submitted guesses.  Each group of candidates (same output grid) is scored as
``wb * agg_guesses(bb - beam) + wa * agg_guesses(agg_augs(ab - s)) + wf * count``.
"""
from __future__ import annotations

import argparse

import numpy as np

from eval_nvarc_offline import load_labels, load_runs


def groups(pool):
    g = {}
    for cand in pool:
        g.setdefault(cand["code"], []).append(cand)
    return g


def _agg(values, how):
    if not values:
        return 0.0
    if how == "sum":
        return float(np.sum(values))
    if how == "mean":
        return float(np.mean(values))
    if how == "max":
        return float(np.max(values))
    if how == "min":
        return float(np.min(values))
    if how == "median":
        return float(np.median(values))
    raise ValueError(how)


def make_scorer(bb, ab, beam_agg, aug_within, aug_across, wb, wa, wf):
    """Rank candidate grids for one task.

    For each candidate grid, per guess: beam = bb - beam_score, and per
    augmentation: s = ab - score_aug.  Within a guess the augmentations are
    aggregated with ``aug_within``; across the guesses of a grid the beam and
    aug terms are aggregated with ``beam_agg``/``aug_across``.  Higher = better.
    """
    def scorer(pool):
        scored = []
        for code, guesses in groups(pool).items():
            beams = [bb - x["beam_score"] for x in guesses]
            augs = []
            for x in guesses:
                aug = x["score_aug"]
                augs.append(_agg([ab - s for s in aug], aug_within) if aug else 0.0)
            total = (
                wb * _agg(beams, beam_agg)
                + wa * _agg(augs, aug_across)
                + wf * len(guesses)
            )
            scored.append((total, code))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [code for _, code in scored]

    return scorer


def build_candidates():
    cands = [
        # faithful replications of the notebook's two selectors
        ("kgmon_ref", make_scorer(0, 0, "mean", "mean", "mean", 0, 1, 1)),
        ("probmul_ref", make_scorer(3, 3, "sum", "sum", "mean", 1, 1, 0)),
        ("freq_only", make_scorer(0, 0, "sum", "mean", "mean", 0, 0, 1)),
    ]
    # structured sweep
    for bb in (3, 4):
        for ab in (3, 4):
            for beam_agg in ("sum", "mean", "max"):
                for aug_within in ("sum", "mean", "median", "min", "max"):
                    for aug_across in ("sum", "mean", "max"):
                        cands.append(
                            (
                                f"p b{bb}/{ab} {beam_agg}|{aug_within}|{aug_across}",
                                make_scorer(
                                    bb, ab, beam_agg, aug_within, aug_across, 1, 1, 0
                                ),
                            )
                        )
    # weighting variants around the faithful structure
    for wb, wa in ((1, 0.25), (1, 0.5), (1, 2), (0.5, 1), (2, 1), (1, 5), (5, 1)):
        cands.append(
            (f"p w{wb}/{wa}", make_scorer(3, 3, "sum", "sum", "mean", wb, wa, 0))
        )
    for wf in (0.25, 0.5, 1, 2, 5):
        cands.append(
            (f"p+freq {wf}", make_scorer(3, 3, "sum", "sum", "mean", 1, 1, wf))
        )
    return cands


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", required=True)
    ap.add_argument("--solutions", default="data/arc-agi_evaluation_solutions.json")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    runs = load_runs(args.store)
    labels = load_labels(args.solutions)
    tasks = {}
    for run in runs.values():
        for key, cands in run.items():
            tasks.setdefault(key, []).extend(cands)
    labelled = {k: v for k, v in tasks.items() if k in labels}
    n = len(labelled)
    if not n:
        raise SystemExit("no labelled tasks found")
    print(f"store={args.store}  labelled tasks={n}")

    results = []
    for name, scorer in build_candidates():
        n1 = n2 = 0
        for key, pool in labelled.items():
            label = labels[key]
            order = scorer(pool)
            if order and order[0] == label:
                n1 += 1
            if label in order[:2]:
                n2 += 1
        results.append((n2 / n, n1 / n, name))
    results.sort(key=lambda r: (r[0], r[1]), reverse=True)

    print(f"{'n2':>7} {'n1':>7}  scorer")
    for n2, n1, name in results[: args.top]:
        print(f"{100 * n2:6.2f}% {100 * n1:6.2f}%  {name}")
    print("--- reference (notebook) selectors ---")
    for n2, n1, name in results:
        if name.endswith("_ref") or name == "freq_only":
            print(f"{100 * n2:6.2f}% {100 * n1:6.2f}%  {name}")


if __name__ == "__main__":
    main()