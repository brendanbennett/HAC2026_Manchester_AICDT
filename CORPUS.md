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

## The result: the prediction was right, and the conclusion is the opposite of the hope

**The gate passed decisively.** At `--eta-scale 0.5` the flow carves model 3 to convexity
**0.6934**, against a pre-registered gate of 0.980 and the flat 1.000 that every previous flow
in this project has produced. The truth's own convexity is 0.7681, so it goes from the wrong
side of the answer to slightly past it, in one change. The diagnosis was right: eta was the
knob that decided whether the network carves at all.

**And carving made everything worse.**

| model | convexity | truth convexity | score | vs convex |
|---|---|---|---|---|
| 1 (Vesta) | 0.8664 | 0.9965 | 1.8697 | -0.10 |
| 2 (sawed-off cube) | **0.4996** | **0.9999** | 1.5850 | -0.32 |
| 3 (Mithra) | 0.6934 | 0.7681 | 1.5113 | -0.16 |
| | | | **4.9660** | **-0.60** |

It carved a **cube** down to half its volume. And on Mithra, where the convexity is now nearly
right, the score still *fell* -- Dice 0.5566 against the convex stage's 0.7147. The carving is
the right size and in the wrong places.

### What eta actually is

eta is not a noise level. It is **how much the network is told to distrust our forward model**,
and the sweep shows the network's convexity was never a defect:

- eta high: the data is discounted, the answer is convex, score 5.48-5.52.
- eta low: the data is trusted, the body carves, it fits the curves better and truth worse,
  score 4.97.

That is the project's central anti-correlation -- misfit against Dice -- arriving from a fifth
independent direction, after MAP descent, the carving search, the GA on another branch, the
widened `dh` band and block alternation. **The flow was correctly discounting a forward model
that an independent renderer beats by 17x.** Its convexity was the right response to the
information it was given, not a failure to learn.

### What this does and does not license

It does not license "the corpus is the problem". The library is diverse and deeply carved
(above), and the concavity signal really is below the injected noise -- both measurements
stand. What is wrong is the inference I drew from them: recovering the signal lets the network
act on the data, and acting on this data is harmful.

The only untested direction the sweep points at is **the other way**: if trusting the data less
is better, the calibrated eta may already be too low. Arms at 1.5 and 2.5 test it. If they are
flat or worse, eta is already optimal and this line is closed.
