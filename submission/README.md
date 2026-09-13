# Submission — the seven scored bodies

`Asteroid04.stl` … `Asteroid10.stl`. These are the files to submit.

**5.5632 / 6** on the public models under the organisers' own evaluation code, against
**5.5442** for the convex stage — the only pipeline in this repo that beats convex.

All seven are watertight, a single component, of positive volume, and posed in the challenge
frame with `r_xy` matching the published cylinder radius (`scripts/check_submission.py`).

## Method, in short

A learned primal-dual convex inversion gives a convex body from the lightcurves, and a
conditional flow carves concavities onto it. Then, per model, the better of those two is chosen
by an **independent forward model**: DAMIT's renderer, scored against the Blender lightcurves
that ship with all ten bodies. The shapes are ours; only the *choice* between them is
outsourced, because our own misfit is anti-correlated with shape accuracy while the referee
picks the higher-scoring body 3 times out of 3 on the public models.

| model | chosen | |
|---|---|---|
| 4, 5, 6, 7, 9, 10 | carved (`results/flow_all`) | the referee preferred the carved body |
| 8 | convex (`results/convex`) | referee preferred convex; model 8 is near-spherical and its phase is unidentifiable |

Reproduce the score, and the selection:

```
PYTHONPATH=. python scripts/benchmark.py results/referee_public results/convex --models 1 2 3
PYTHONPATH=. python scripts/select_by_referee.py --out /tmp/check   # same picks
```

## Caveats

The 5.5632 is **three public bodies, two of them near-convex**. Nothing here is measured on
models 4–10, and the selection rule is validated 3/3 — but that is 3.

`results/submission_convex_fallback/` is the conservative alternative (convex everywhere,
5.5442). It differs from this on six of the seven bodies. `BENCHMARKS.md` has the full
scoreboard and every method that was tried and rejected.
