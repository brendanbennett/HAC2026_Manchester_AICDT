# Changes in this archive (2026-08-19)

## 1. The bug: `reconstruct_lpd.py` never imposed the published cylinder radius

`R = CYLINDER_R[a.model]` was read at the top and handed to the solver, but the medoid mesh
went straight to `export_stl`. The convex branch does apply `fit_to_cylinder` (through
`eval_exact.decode`); the LPD branch did not. Measured ratio of achieved radius to
published R:

| model | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| convex | 0.99 | 0.99 | 0.98 | 0.99 | 0.97 | 0.86 | 0.84 | 0.99 | 0.94 | 0.97 |
| lpd | 0.90 | **0.59** | 1.01 | **0.65** | 0.76 | 0.90 | 0.76 | 0.77 | 0.98 | **0.22** |

For a body scaled by `s` in xy and otherwise contained in the truth, Dice = 2s²/(1+s²).
At s = 0.59 the score is capped at 0.52 however good the shape is — model 2 measured
0.4988. Model 10 at s = 0.22 is capped at 0.09.

**Fixed permanently** in `scripts/reconstruct_lpd.py`: the mesh is now posed and scaled to
`--cylinder-fill` × R (default 0.99) before export. `--no-fit-cylinder` disables it for
ablations. The 0.99 is because the published R is a tight upper bound, not an equality:
the public models reach 0.98–0.99 of it.

## 2. Effect, measured against the public STLs (`--fast` settings)

```
                  model 1   model 2   model 3   summed voxel
lpd (as it was)    0.8799    0.4988    0.6680      2.0468
lpd_fitted         0.9727    0.9346    0.6836      2.5909
convex (submitted) 0.9792    0.9128    0.6922      2.5843
```

**+0.54 summed on three models from one multiplication.** Re-posed LPD now edges out the
submitted convex set overall (2.5909 vs 2.5843), driven entirely by model 2.

Two things this exposes, both more important than the corpus work:

- **On model 3 the flow is still slightly worse than the convex solver** (0.6836 vs
  0.6922). The non-convex correction is currently not paying for itself on the only
  non-convex model we can check.
- **Model 3's convex reconstruction is itself far from the achievable convex answer.**
  The true model 3's own convex hull scores 0.852 against it; the convex stage is
  delivering 0.692. So there is ~0.16 of headroom on model 3 *before any concavity
  modelling at all*, which is about the same size as the entire remaining gap from the
  hull to a perfect reconstruction (0.852 → 1.0). Worth attacking first.

Also flagged: **convex models 6 and 7 sit at 0.86 and 0.84 of R** while everything else is
at 0.97–0.99. Those two are probably losing a few points to the same width issue and are
worth a look.

## 3. New files

- `scripts/fix_pose.py` — re-poses existing STLs onto the published cylinder. Used to
  produce `results/lpd_fitted/` (all ten models, included). Post-hoc fix; new runs get it
  automatically from the change above.
- `hac26/shapes_nonconvex.py` — the implicit, hull-deficit-stratified training corpus.
  Run `python -m hac26.shapes_nonconvex --n 40` to sample and print the strata report.
- `scripts/train_surrogate.py` — `shapes()` is now a thin wrapper over `sample_corpus`.
  Feature cache key bumped `v7_` → `v9_` so a stale cache cannot silently retrain on the
  old corpus.
- `runs/gate_{lpd,lpd_fitted,convex}.json` — the numbers above, as produced by
  `eval_gate.py`.

## 4. Suggested order from here

1. Re-run `reconstruct_lpd.py` for all ten models with the fix in place (or just use
   `results/lpd_fitted/`, which is equivalent for the already-computed medoids).
2. Check convex models 6 and 7's radius shortfall.
3. Look at why the convex stage only reaches 0.692 on model 3 against an 0.852 ceiling.
4. Only then retrain with the new corpus and run the gate — baseline is now
   `runs/gate_lpd_fitted.json`, not the unfitted one.
