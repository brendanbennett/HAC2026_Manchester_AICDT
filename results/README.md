# Which directory is the submission

**The submission now lives at the top level of the repo: `submission/`** — Asteroid04.stl … Asteroid10.stl, with its own README.

It used to sit here as `results/SUBMISSION/`; it was moved out so it is the first thing anyone sees.

Scores **5.5632 / 6** on the public models with the organisers' own evaluation code, against
**5.5442** for the convex stage. It is the only pipeline in this repo that beats convex.

Reproduce the number (the same rule applied to the three public bodies, where truth exists):

```
PYTHONPATH=. python scripts/benchmark.py results/referee_public results/convex --models 1 2 3
```

## What is in it

Per body, the better of two candidates, chosen by an independent renderer — DAMIT's, validated
against the Blender reference curves to 0.008 RMSE on a truth mesh — scoring each against the
Blender lightcurves that ship with all ten models. The shapes are ours; only the *choice*
between them comes from the referee. It picks the higher-scoring body 3 times out of 3 on the
public models for the convex-vs-carved question.

| model | source | why |
|---|---|---|
| 4, 5, 6, 7, 9, 10 | `flow_all/` (carved) | referee preferred the carved body |
| 8 | `convex/` | referee preferred convex; model 8 is near-spherical and its phase is unidentifiable |

`scripts/select_by_referee.py` reproduces the selection; `BENCHMARKS.md` has the full scoreboard
and the method.

## The other directories

| directory | what | ships? |
|---|---|---|
| **`../submission/`** | referee-selected, 5.5632 | **yes — at the repo root** |
| `submission_convex_fallback/` | convex stage only, 5.5442 | no — the safe fallback, kept intact |
| `convex/` | LPD convex stage, all ten | no — an input to the selection |
| `flow_all/` | 800-body flow, all ten | no — the other input to the selection |
| `final_p30/`, `flow_p0/`, `flow_p30/`, `lpd/` | earlier pipelines, all scored in BENCHMARKS.md | no |
| `referee_public/`, `referee_all/` | the same rule on models 1–3 / all ten, for validation | no |
| `figures/` | renders and training plots | — |

## Caveats, stated plainly

The 5.5632 is **three public bodies, two of them near-convex**, so it is a thin validation set.
Nothing here is measured on models 4–10. The selection rule is validated 3/3 but that is 3.
