# Results

What the two reconstruction methods in this repository score, and why the convex stage is
the one submitted.

The challenge scores a reconstruction two ways: a voxel overlap measure, higher is better
and 1 is perfect, and a side-view distance between the boundary curves of 2D projections,
lower is better. Only asteroids 1-3 are public, so those are the only real bodies either
method can be scored against here; asteroids 4-10 are the scored, secret ones.

## The public asteroids

Reproduced by `make submission` and scored by `hac26/scoring/voxel.py` and
`hac26/scoring/side_view.py`.

| model | convex voxel | flow voxel | convex side ASSD | flow side ASSD |
| --- | --- | --- | --- | --- |
| 1 | **0.9800** | 0.7528 | **0.0122** | 0.0742 |
| 2 | **0.8975** | 0.7008 | **0.0671** | 0.1394 |
| 3 | **0.6914** | 0.5412 | **0.1139** | 0.1406 |
| sum / mean | **2.5689** | 1.9948 | **0.0644** | 0.1181 |

> The flow column is the flow trained without the rollout phase, sampled at guidance 1.0.
> The final flow — rollout-trained, guidance 2.0 — rewrote `results/lpd` but was never
> scored against it, because the `score` stage's cache key did not cover the settings that
> changed. That is fixed in `scripts/run_remote_pipeline.sh`; rescoring is a
> `rm runs/.done/score` and about ninety seconds. On the corpus the two differ by roughly
> +0.017 Dice, which does not approach the 0.19 gap below, so the conclusion is not expected
> to move — but it is measured for one flow and asserted for the other.

The convex stage wins on every model under both measures. Two reference rows put that in
context, both from the side-view scorer:

| model | truth vs itself | convex hull of the truth |
| --- | --- | --- |
| 1 | 0.0003 | 0.0011 |
| 2 | 0.0004 | 0.0004 |
| 3 | 0.0002 | 0.0386 |

The first is the floor set by sampling and pixel resolution. The second is what a perfect
convex reconstruction would score, so it says how non-convex each body really is: asteroids
1 and 2 are nearly convex, and asteroid 3 is the only public body with a carve worth
recovering. It also shows the convex stage is not at its own ceiling — 0.0122 against a
possible 0.0011 on model 1 — so there is room in the convex stage independent of the flow.

## Held-out corpus bodies

The flow was selected against synthetic bodies from the training corpus, reconstructed
exactly as the challenge models are (`scripts/decision_check.py`, four held-out bodies,
`runs/decision_check.json`). There it does slightly better than the convex start it
corrects:

| guidance | flow Dice | convex Dice | gain |
| --- | --- | --- | --- |
| 1.0 | 0.7192 | 0.7262 | -0.0070 |
| **2.0** | **0.7357** | 0.7262 | **+0.0095** |
| 3.0 | 0.7304 | 0.7262 | +0.0042 |

Split by how carved the body is, the picture is consistent across every guidance weight:

| guidance | low carve | medium carve | high carve |
| --- | --- | --- | --- |
| 1.0 | -0.0199 | -0.0445 | +0.0181 |
| 2.0 | -0.0019 | -0.0146 | +0.0273 |
| 3.0 | -0.0081 | -0.0149 | +0.0198 |

The flow adds carving. That helps a genuinely non-convex body and costs a nearly convex one,
which is what these numbers say and is the behaviour the method is designed to have.

## Why the convex stage is submitted

The flow gains +0.0095 Dice on synthetic held-out bodies and loses 0.19 on the real public
ones. The gap is not explained by which bodies are carved: it loses on asteroid 3 as well,
the one public body where the carve-bin table predicts it should win.

The most likely reading is a mismatch between the corpus and reality. The corpus spans
bodies carved 0.05 to 0.51, while the public asteroids are far closer to convex, so a flow
calibrated to the corpus over-carves a real asteroid. The convex stage has no such exposure:
it never carves, and on bodies that are nearly convex to begin with that is the right bias.

Submitting the convex stage is therefore the choice the evidence supports, and
`make submission` builds it by default. `METHOD=flow make submission` builds the other.

## Reproducing

```
make venv toolchain
make data
make submission        # the ten submitted STLs into results/convex
```

Scoring the public models needs the released data and the truth STLs it contains:

```
PYTHONPATH=. python hac26/scoring/voxel.py --stl results/convex/Asteroid0{1,2,3}.stl --models 1 2 3
PYTHONPATH=. python hac26/scoring/side_view.py --models 1 2 3 --recon-dir results/convex
```
