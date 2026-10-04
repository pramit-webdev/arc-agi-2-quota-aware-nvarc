# ARC Prize 2026 — ARC-AGI-2

A quota-aware pipeline for the **ARC Prize 2026 - ARC-AGI-2** Kaggle
competition, built on the public **NVARC** lineage (see `THIRD_PARTY.md`), with
offline tooling for improving the answer-selection step without spending GPU
quota.

## Why this exists

The competition is brutally compute-limited:

- one scored submission per day;
- a maximum notebook runtime of 12 hours;
- **L4x4 machines consume GPU quota at twice the T4x2/P100 rate**, and the
  floating weekly quota is ~30 hours — so a single full run costs ~24 quota
  hours, i.e. roughly one run per week.

Under those limits the score is **coverage-limited**: for most tasks the model
never generates the correct grid, so no selector can recover it. This repository
therefore (1) measures coverage and selector accuracy offline on captured
candidate pools, (2) improves the selector, and (3) documents the quota math so
the one expensive run per week is spent well.

## Contents

| Path | Purpose |
| --- | --- |
| `build_nvarc_notebook.py` | Builds the notebooks. `--harvest` builds the dev variant; `--budget-hours` sets the search budget. |
| `notebooks/kaggle_nvarc_lb33/` | Submission candidate (L4x4, 12h budget, robust selector). |
| `notebooks/kaggle_nvarc_harvest/` | Dev notebook: runs the full public evaluation set and captures candidate pools. |
| `notebooks/kaggle_nvarc_t4x2/` | T4x2 (4-bit) variant for cheaper runs and ensembling. |
| `eval_nvarc_offline.py` | Offline harness: scores selection strategies against labelled pools (numpy only). |
| `tune_nvarc_picker.py` | Grid search over selector ("picker") formulas. |
| `THIRD_PARTY.md` | Upstream attribution, licensing, verified competition constraints, and results. |
| `RESULTS.md` | Full offline study numbers (coverage, selector accuracy, CV, CIs). |
| `SOLUTION_WRITEUP.md` | Writeup addressing the six Innovation-Prize criteria. |
| `watch_and_submit_lb33.sh` | Watches the queued submission version and submits it once it completes. |

## The pipeline in one paragraph

The upstream notebook fine-tunes a small grid-token LLM per task, runs an
ARC-token DFS beam search over 16 augmented views, scores every candidate grid
under 16 further augmentations, and writes the two best guesses per test input.
This repository keeps that core and adds: a GPU-adaptive worker count, the full
12-hour budget, a **candidate-pool exporter** (so a run's search results can be
re-analysed offline), an **offline picker tuner**, and a **robust selector**
that ranks candidate grids by DFS beam score plus the *median* augmented score.

## Reproduce

```bash
# build the submission notebook (12h budget) and the dev harvest notebook (3h)
python3 build_nvarc_notebook.py
python3 build_nvarc_notebook.py --harvest --budget-hours 3

# push them to Kaggle
source ./kaggle_auth.sh
kaggle kernels push -p notebooks/kaggle_nvarc_harvest   # dev: captures pools
kaggle kernels push -p notebooks/kaggle_nvarc_lb33      # submission candidate

# after a harvest run finishes, download and analyse the pools locally
kaggle kernels output <owner>/arc-2026-agi2-nvarc-harvest-v1 -p /tmp/harvest
python3 eval_nvarc_offline.py --store /tmp/harvest/inference_outputs
python3 tune_nvarc_picker.py  --store /tmp/harvest/inference_outputs

# submit the (completed) notebook version to the code competition
# NOTE: -f submission.json is REQUIRED; omitting it returns a 400 Bad Request.
kaggle competitions submit arc-prize-2026-arc-agi-2 \
  -k <owner>/arc-2026-agi2-nvarc-lb33-v1 -v <version> -f submission.json \
  -m "nvarc lb33 + robust selector, 12h budget"

# or let the watcher wait out the L4x4 queue and submit automatically:
./watch_and_submit_lb33.sh
```

## License

Original work: **CC BY 4.0** (see `LICENSE`). Third-party components retain
their own terms (`THIRD_PARTY.md`).
