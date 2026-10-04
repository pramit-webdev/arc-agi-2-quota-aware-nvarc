# Offline study results

All numbers below come from one **3-hour harvest run** of the public evaluation
set (kernel `pramitdas/arc-2026-agi2-nvarc-harvest-v1`), whose candidate pools
were captured and scored locally with `eval_nvarc_offline.py` and
`tune_nvarc_picker.py`. Reproduce with the commands in `README.md`.

**Pool:** 1,330 candidate grids across **92 labelled tasks** (avg 14.5
candidates/task).

## The headline: coverage limits everything

| Quantity | Value |
| --- | --- |
| Pool coverage (correct grid somewhere in the pool) | **36.96%** (34/92) |
| Coverage @ top-2 of the *best* selector | 29.35% |
| Coverage @ top-8 (any selector) | 37.0% |

Even a perfect selector cannot exceed **36.96%** on this pool: for the other
63% the model never generated the correct grid. **Selection is a second-order
lever; search coverage is first-order.**

## Selector accuracy (n_guess=2, the competition metric)

| Scorer | n1 | n2 |
| --- | --- | --- |
| `beam` (DFS beam score only) | 18.48% | 25.00% |
| `uniform` (candidate frequency) | 20.65% | 25.00% |
| `kgmon` (stock upstream selector) | 20.65% | 27.17% |
| `probmul` (`getter_full_probmul_3`) | 21.74% | 28.26% |
| **`score_robust`** (deployed: `b4/sum\|median\|sum`) | **22.83%** | **29.35%** |

`score_robust` groups candidates by proposed grid and scores each group by

```
sum_g (4.0 - beam_score_g)  +  sum_g median_aug (4.0 - score_aug_g)
```

i.e. beam score plus the **median** (rather than mean) over the 16 scoring
augmentations — so a few corrupt augmentations cannot swing a grid's rank. It
is the unique top scorer in the `tune_nvarc_picker.py` grid search and never
loses to `kgmon` in 5-fold cross-validation.

## Cross-validation (5 folds over 92 tasks)

| Scorer | fold0 | fold1 | fold2 | fold3 | fold4 | mean |
| --- | --- | --- | --- | --- | --- | --- |
| `kgmon` (default) | 21.1 | 21.1 | 22.2 | 22.2 | 50.0 | 27.17 |
| `probmul` | 31.6 | 21.1 | 16.7 | 22.2 | 50.0 | 28.26 |
| **`score_robust`** | 31.6 | 21.1 | 22.2 | 22.2 | 50.0 | **29.35** |

`score_robust` is `>=` both stock selectors in every fold.

## Significance (bootstrap 95% CI of the gain vs `kgmon`)

| Comparison | Δ (pts) | 95% CI |
| --- | --- | --- |
| `probmul` vs `kgmon` | +1.09 | [-2.17, +4.35] |
| **`score_robust` vs `kgmon`** | **+2.17** | **[+0.00, +5.43]** |

The gain is small and only **borderline** significant on 92 tasks. It is
reported as such: it is a cheap, safe, weakly-dominant improvement, not a
breakthrough. The real finding is the coverage ceiling.

## Quota arithmetic (why this matters)

Under ARC Prize 2026 rules a notebook run costs GPU quota, and L4x4 machines
consume it at **2×** the T4x2/P100 rate; the floating weekly quota is ~30h.
So one full 12-hour L4x4 run costs ~24 quota-hours — roughly **one big run per
week**, and at most **one scored submission per day**. The harvest → offline
tune → single-best-run loop exists precisely to make that one run count.