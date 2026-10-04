# Third-party provenance

## NVARC candidate notebook

- Upstream Kaggle notebook: `mikelou1/arc-agi2-lb33-89-minimal-perfpatch`
- Upstream URL: https://www.kaggle.com/code/mikelou1/arc-agi2-lb33-89-minimal-perfpatch
- Upstream model: `sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1`
- Upstream environment patch: `sorokin/pip-install-unsloth-flash-patch`
- Official 2025 results: https://arcprize.org/competitions/2025
- Preserved source SHA-256: `ee49384a10a4296dc0bf0670ae67325521f798d5337ed57bee50c8a443a378f4`

The exact pulled notebook is retained as
`notebooks/kaggle_nvarc_lb33/upstream_source.ipynb`. The generated notebook
adds attribution and quota-safety changes documented in
`build_nvarc_notebook.py`.

The pulled Kaggle notebook does not expose an explicit software license in its
source or metadata. It is used here as an attributed public competition
baseline. Verify upstream licensing and competition prize-eligibility terms
before claiming this derived implementation as prize-eligible open source.

## NVARC T4x2 candidate notebook

- Upstream Kaggle notebook: `nihilisticneuralnet/baseline-nvarc-arc-25-winning-solution-for-t4x2`
- Upstream URL: https://www.kaggle.com/code/nihilisticneuralnet/baseline-nvarc-arc-25-winning-solution-for-t4x2
- Upstream model: `sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1`
- Upstream environment patch: `sorokin/pip-install-unsloth-flash-patch`
- Preserved source SHA-256: `452dbb1fc050bad9cbba38a6802231bff91cc2970899eabb8c74e5f34f322a6c`

The exact pulled notebook is retained as
`notebooks/kaggle_nvarc_t4x2/upstream_source.ipynb`. The generated notebook
adds attribution and a 6.5-hour quota-safe budget.

The pulled Kaggle notebook does not expose an explicit software license in its
source or metadata. It is used here as an attributed public competition
baseline. Verify upstream licensing and competition prize-eligibility terms
before claiming this derived implementation as prize-eligible open source.

## NVARC harvest-mode dev notebook

- Builder: `python3 build_nvarc_notebook.py --harvest`
- Output dir: `notebooks/kaggle_nvarc_harvest/` (kernel id
  `pramitdas/arc-2026-agi2-nvarc-harvest-v1`).
- Same preserved upstream source as the LB33 candidate, with two dev-only
  changes: the starter run cell sets `ARC_HARVEST=1` so a Save & Run All run
  queues the full 120-task public evaluation set, and candidate pools are
  written to `/kaggle/working/inference_outputs` (`ARC_OUTPUTS_DIR`
  overridable) so they survive in the captured kernel output.
- Development only. Never submit this notebook to the competition; submit the
  LB33 candidate (`pramitdas/arc-2026-agi2-nvarc-lb33-v1`).

## Offline pool evaluation

- `eval_nvarc_offline.py` consumes the harvested pool pickles and benchmarks
  retrieval/aggregation strategies (`beam`, `probmul`, `kgmon`, `uniform`) and
  two-guess combinations against `data/arc-agi_evaluation_solutions.json`.
- Pure numpy, so it runs on the 4 GB local workstation. It reports pool
  coverage, best-order rank curves, and n_guess=1/2 accuracy.

## Offline picker tuning

- `tune_nvarc_picker.py` grid-searches selector ("picker") formulas against the
  harvested pools; `eval_nvarc_offline.py` reports coverage and per-strategy
  accuracy.
- First harvest (92 labelled tasks, 3-hour run): pool coverage (the correct grid
  is somewhere in the candidate pool) is **36.96%**.  The stock `score_kgmon`
  selector used by the upstream notebook scores **27.17%** (n_guess=2);
  `probmul` scores **28.26%**; a robust `score_robust` (beam score plus the
  **median** augmented score) scores **29.35%**.
- 5-fold cross-validation shows `score_robust` weakly dominates the stock
  selector (never worse in any fold).  Bootstrap 95% CI for the gain is roughly
  **[+0.0, +5.4] points**, so the improvement is small and only borderline
  significant on 92 tasks.
- The builder therefore replaces the stock `run_selection_algo()` (kgmon) with
  `run_selection_algo(score_robust)`.  Revert by removing that patch if a larger
  harvest contradicts it.

## Kaggle account (competition-eligibility)

The kernels are pushed under a single Kaggle account. The competition rules
state: *"You cannot sign up to Kaggle from multiple accounts and therefore you
cannot enter or submit from multiple accounts"*, and *"You will be disqualified
if you make Submissions through more than one Kaggle account."*

- Active account: **`pramitdas`** (the only account, already joined to the
  competition). Kaggle declined the attempt to create a second account, so the
  single-account rule is satisfied.
- Override the owner at build time with `ARC_KAGGLE_OWNER=<name>` if needed.

## Competition constraints (verified from the official pages, 2026-10-02)

Sources: `arcprize.org/competitions/2026`, the Kaggle rules page, and the
competition Overview/Evaluation page.

- Submissions: a maximum of **one (1) scored Submission per day**.
- Notebook runtime: CPU and GPU notebooks must run in **<= 12 hours**.
- Final standing is the **Private Leaderboard**; the Public Leaderboard only
  reflects the public test set.
- L4x4 machines "consume GPU quota at twice the rate of the older T4x2 and P100
  machines". The floating weekly quota is 30 hours (sometimes higher), so a
  12-hour L4x4 run costs about **24 quota hours**, i.e. roughly one such run per
  week.
- Score is coverage-limited: the upstream 12-hour budget processes roughly 192
  of 240 test tasks, which is what produced the 33.89 baseline. Shortening the
  budget cuts coverage and the score roughly proportionally, so the builder
  keeps the full upstream 12-hour budget (`SEARCH_BUDGET_LINE`).

