# Solution Writeup — ARC Prize 2026, ARC-AGI-2

**Author:** Pramit Das · **Kaggle:** `pramitdas`
**Track:** ARC-AGI-2 · **Code:** https://github.com/pramit-webdev/arc-agi-2-quota-aware-nvarc (CC BY 4.0)
**Notebook:** `pramitdas/arc-2026-agi2-nvarc-lb33-v1`

## 1. Summary

This submission competes under ARC Prize 2026's hard compute limits, where the
binding constraint is not the model but **quota**. We contribute a
**measurement-driven workflow** — harvest the model's candidate pools on the
public evaluation set, tune the answer selector offline at zero GPU cost, then
spend the single expensive run per week on the best-known configuration — plus a
small but robust **selector improvement**.

## 2. The system

The core is the public NVARC lineage (`sorokin/qwen3_4b_grids15_sft139` with a
per-task test-time fine-tune and an ARC-token DFS beam search). Each task is
solved independently: (1) the attached 4B grid-token LLM is LoRA fine-tuned on
the task's 128 augmented demonstrations; (2) a DFS beam search over the ARC token
vocabulary produces candidate grids under 16 augmented views; (3) every candidate
is re-scored under 16 further augmentations ("augmented score"); (4) candidates
are grouped by their proposed grid and the top two become `attempt_1` /
`attempt_2`.

Our substantive changes to this lineage (all listed in `THIRD_PARTY.md`) are:

- **Candidate-pool export.** The upstream pipeline writes its per-task candidate
  pools to a path discarded after the run. We redirect them into
  `/kaggle/working` so a run's entire search output survives as reusable data.
- **Robust selector.** The stock selector (`kgmon`) is replaced by
  `score_robust`, scoring a candidate grid by the sum over its guesses of
  `(baseline - beam_score)` plus the sum over its guesses of the **median** over
  augmentations of `(baseline - score_aug)` — the median reduces sensitivity to
  outlier augmentations.

## 3. What the offline study found

A 3-hour harvest run on the public evaluation set produced 1,330 candidate grids
across 92 labelled tasks (full tables in `RESULTS.md`). Against the labels:

- **Pool coverage** (correct grid somewhere in the pool): **36.96%**.
- Selector accuracy (`n_guess=2`, the competition metric): `kgmon` **27.17%**,
  `probmul` **28.26%**, `score_robust` **29.35%**.

The headline result is the **coverage ceiling**: even a perfect selector cannot
exceed ~37% on these tasks, because the model never generated the correct grid
for the other ~63%. **Selection is a second-order lever; search coverage is the
first-order one.** A materially higher score therefore needs more/better search
(longer runs, ensembling pools), not a cleverer picker.

## 4. The six Innovation-Prize criteria

**Accuracy.** Reproduces the NVARC baseline's leaderboard level (~33.9) with a
small selector gain measured offline; reported honestly, not overstated.

**Universality.** The workflow is not ARC-specific. Any system that (a) does
per-task test-time search and (b) ranks candidate outputs can reuse the loop:
capture the candidate pool once, then tune the ranking offline against a
labelled development set at zero GPU cost. The offline tools are generic over
pools of `{candidate, score_vector}` records.

**Progress.** Contributes a concrete, reusable result that redirects effort:
under the competition's own constraints the score is **coverage-limited (~37%
ceiling in our measurement)**, so teams chasing the 85% Bonus should invest in
search/coverage and model quality, not selection heuristics. We also document
the quota arithmetic (L4x4 = 2× rate ⇒ ~24 quota-hours per 12-hour run) that
governs how many experiments a solo entrant can afford.

**Theory.** A task is solved iff the correct grid appears in the candidate pool
*and* the selector ranks it in the top two. The pool comes from a beam search
whose recall is bounded by the fine-tuned model's per-token accuracy and the
search budget; the selector only re-orders what the search found. Because
augmented scoring can be corrupted by a few bad augmentations, a **median**
(rather than mean) aggregation of augmented scores is the more robust estimator
of a grid's true fit — which is why `score_robust` weakly dominates the stock
mean-based selector in cross-validation.

**Completeness.** The leaderboard submission is produced end-to-end by the open
notebook, and every supporting artifact is released: builders, the harvest and
offline-evaluation tools, the selector tuner, the provenance/licensing record,
and this writeup. Reproduction commands are in `README.md`.

**Novelty.** Modest and honestly scoped. The model, search, and two base
selectors are upstream NVARC work. Our contributions are the quota-aware
harvest/offline-tune workflow, the candidate-pool export, the robust
median-augmented selector, and the coverage-limit analysis. We claim no new
architecture.

## 5. Limitations (stated plainly)

- The selector gain is **small and only borderline significant** on 92 tasks
  (bootstrap 95% CI ≈ [+0.0, +5.4] points).
- The coverage figure comes from a **3-hour** harvest; a full 12-hour run covers
  more tasks and may shift the numbers.
- We did not attempt a new model or architecture; this lineage's ceiling is far
  below the leading teams.

## 6. Artifacts

- Repository (open source, CC BY 4.0):
  https://github.com/pramit-webdev/arc-agi-2-quota-aware-nvarc
- Submission notebook: `pramitdas/arc-2026-agi2-nvarc-lb33-v1`
- Dev harvest notebook: `pramitdas/arc-2026-agi2-nvarc-harvest-v1`
- In the repository: notebook builders, `eval_nvarc_offline.py`,
  `tune_nvarc_picker.py`, `RESULTS.md` (all measured numbers), `THIRD_PARTY.md`,
  `LICENSE` (CC BY 4.0).
