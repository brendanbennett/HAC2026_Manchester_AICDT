# The corpus is fine. The noise we inject into it is not.

## What the corpus actually contains

The assumption that the library is "convex shapes with craters added" is wrong, and it is
worth correcting before anyone spends time regenerating it. The 2500-body library built for
the 09-12 run (`logs/csf3_20199623.out`) contains:

| family | bodies | | convexity band | bodies |
|---|---:|---|---|---:|
| bilobe (contact binary) | 614 | | < 0.55 | 320 |
| object (Thingi10K prints) | 492 | | 0.55 - 0.70 | 702 |
| real asteroid models | 306 | | 0.70 - 0.85 | 731 |
| potato | 271 | | 0.85 - 0.95 | 443 |
| geometric (saw cuts) | 267 | | >= 0.95 | 304 |
| trilobe | 264 | | | |
| faceted | 192 | | pairwise Dice | 0.534 mean |
| top (equatorial ridge) | 94 | | PR(combined) | 15.49 |

**35% are contact binaries, 70% have convexity below 0.85, and 306 are real asteroid shape
models.** The TLS failure recorded in the original plan -- which silently dropped the `real`
and `object` families, 26% of the family weight -- was fixed before this run: the log reads
"28 real shape models, 191 objects". The library is diverse, deeply carved, and drawn from the
same sources as the public bodies.

Regenerating it is not where the problem is.

## The concavity signal is smaller than the noise we add to it

Training does not use the corpus curves as rendered. It uses

    data = curves + sigma * xi + eta * zeta

with `sigma` the measurement noise (0.0005 - 0.004, tiny) and **`eta` the calibration's fitted
model error, injected as smooth random noise shaped like the curve**. For the 09-12 run
eta = 0.1188.

Measure what a concavity is worth in the curves. Scoring the released truth against its own
convex hull with an independent renderer, on the Blender curves:

| model | truth | its convex hull | concavity signal |
|---|---|---|---|
| 3 (Mithra, contact binary) | 0.0172 | 0.0951 | **0.078** |
| 1 (Vesta, near-convex) | 0.0095 | 0.0102 | 0.0007 |

**Mithra's entire concavity signal is 0.078 and we inject 0.119 of noise on top of it.**
Signal-to-noise 0.66. The concavity sits *below* the noise the training distribution declares
is there, and the statistically correct response to such a distribution is to ignore
concavity-scale structure and answer convex. That is what every flow trained here has done:
the 2500-body model emits convexity 0.982-1.000 on all ten models, and zeroing its carving
block changes the public score by 0.0025.

The network is not failing to learn. It is correctly learning what we told it.

## Why this is self-inflicted

`eta` is not a property of the measurement. It is **our renderer's disagreement with the
measurement**, fitted by `scripts/calibrate.py` to absorb whatever the forward model cannot
reproduce. DAMIT's independent renderer reaches ~0.005 RMSE against the same Blender curves on
a truth mesh, where ours needs eta = 0.086 (Blender) or 0.119 (real). So we measured a
deficiency of our own forward model and then injected it into training at full strength, as if
it were irreducible noise on the data.

Worse, it is injected as *random* smooth noise. The real error is systematic and
shape-dependent -- our renderer mis-handles flat facets and self-shadowing in particular ways.
Random noise of that size destroys the concavity information; a systematic bias of that size
would at least be partly learnable.

## The fix under test

`scripts/train_lpd.py --eta-scale` multiplies eta before injection. At 0.5, eta = 0.059 and
Mithra's concavity sits at signal-to-noise 1.3 instead of 0.66.

This trades robustness for signal, and the trade may not pay: a network trained on cleaner
curves than it will meet could carve confidently and wrongly. That is exactly what the public
models are for. Runs 20312571 (0.5) and 20312572 (0.25) are the test.

**Prediction, recorded before the result:** if this diagnosis is right, the reconstructions
should stop being convex -- convexity below 0.98 on model 3, where every previous flow gave
1.000 -- whether or not the score improves. If convexity stays pinned at 1.000, the diagnosis
is wrong and the corpus is not the lever.
