# Scoreboard — public models 1–3

Every claim about a reconstruction belongs here as a row. Run:

```
python scripts/benchmark.py results/convex results/lpd
python scripts/benchmark.py <dir> --models 1 2 3 --pitch 0.05 --n-dirs 4 --label "..."
```

Scored with the organisers' own code (`dataset/raw/Evaluation_measures/`), transcribed and
ported in `hac26/scoring/official.py`: the Python voxel measure verbatim, the MATLAB
projection measure ported. Per model the challenge sums one voxel and one projection score,
each in [0, 1] — 2 per model, **6 is the maximum observable here**, 14 over the seven secret
models. Truth STLs are posed into the challenge frame first (`centre_xy=False`); the released
ones are at the physical scale of the printed model, z spanning 6–8 rather than 2.

## Leaderboard

| date | pipeline | m1 vox | m1 proj | m2 vox | m2 proj | m3 vox | m3 proj | **total** |
|---|---|---|---|---|---|---|---|---|
| 09-08 | *convex hull of truth* — the convex ceiling, not a legal run | 0.9969 | 0.9941 | 0.9997 | 0.9165 | 0.8828 | 0.9848 | **5.775** |
| 09-08 | `results/convex` — LPD convex stage, `--fit-cylinder` | 0.9786 | 0.9929 | 0.9106 | 0.9905 | 0.7147 | 0.9568 | **5.544** |
| 09-08 | `results/lpd` — convex start + flow + polish (as shipped) | 0.9554 | 0.9865 | 0.8109 | 0.9711 | 0.7216 | 0.9589 | **5.405** |
| 09-12 | flow retrained on CSF3, 800 bodies, 11M, polish 30 | 0.9552 | 0.9914 | 0.8598 | 0.9824 | 0.7330 | 0.9576 | **5.480** |
| 09-12 | the same, polish 0 | 0.9530 | 0.9902 | 0.8736 | 0.9764 | 0.7249 | 0.9566 | **5.475** |
| 09-12 | flow retrained, 2500 bodies, **304M** parameters | 0.529\* | — | 0.356\* | — | 0.422\* | — | **far worse** |

\* Dice against truth, not the full measure; the run was abandoned once these were seen.

Model 1 is Vesta (near-convex), 2 the sawed-off cube (convex), 3 Mithra (a contact binary, the
only public body whose shape needs concavity).

## What these numbers say

**The flow costs 0.14 and buys nothing.** The convex stage it is supposed to improve beats it
by 0.116 on voxel and 0.024 on projection. It gains 0.007 on Mithra — the one body where
non-convexity is the whole problem — and loses 0.023 and 0.100 on the two convex ones. Read
with the run logs (369 training steps on a 169-body corpus, validation diverged 17×), the
non-convex correction never fired: three of ten answers are exactly convex and all ten are
≥0.80 Dice to their own convex start.

**The competition is decided on the voxel measure alone.** Projection sits at 0.96–0.99 for
everything, including bodies whose voxel score is 0.71. It spans 0.024 between our best and
worst pipeline while voxel spans 0.116. Optimise Dice; do not spend time on the outline.

**The convex ceiling is not the limit people assume.** Mithra's own convex hull scores 0.883,
only 0.168 short of perfect, and the convex stage reaches 0.715 of that — so two thirds of what
is missing on the hardest public body is *convex-inversion* error, not concavity. Recovering
the waist is worth the remaining third.

**The representation is not the bottleneck.** Fitting the 1728-site Gaussian lattice by exact
least squares with `h` pinned to the true hull — the best the parametrisation can ever do —
reproduces Mithra to Dice **0.991**, a torus to **0.995** and a limbed lego-like figure to
**0.860** (parity-scan Dice at res 96; the official voxeliser reads ~0.02 higher on model 3).
Whatever is losing 0.28 of score, it is not the shape parametrisation.

## What has been tried against the convex answer, and has failed

Everything below beats the convex answer on *misfit* and loses to it on *Dice*, which is the
single most useful fact in this file. The convex stage's answer is at χ 2.1473 and Dice 0.6902
on model 3, against an oracle -- the lattice fitted to the true body on that same convex
support -- at **χ 1.883, Dice 0.9909**. The representation reaches the truth. The objective
prefers the truth to the convex start. And nothing that searches the objective finds it.

| attempt | best χ | Dice there | verdict |
|---|---|---|---|
| convex answer (`results/convex`) | 2.1473 | 0.6902 | the floor |
| *oracle*: lattice least-squares fit to truth, same support | **1.883** | **0.9909** | reachable, not findable |
| Track A: descent on the exact misfit, 1728 amplitudes | 1.109 | 0.665 | fits model error |
| carve search, mixed families, 250 candidates | 2.092 | 0.696 | +2.6%, below the accept margin |
| carve search, 4-parameter waist family, 400 candidates | 2.358 | 0.695 | not one candidate beat convex |
| convex-stage support smoothing, `--smooth 1/2/3` | — | — | 5.527 / 5.484 / 5.153 against 5.543 |

**Why they fail is one fact, measured three ways.** Walk the straight line from the convex
answer to the oracle: χ goes 2.147 → 2.336 at t = 0.4 → 1.898 at t = 1, while Dice climbs
0.690 → 0.991 the whole way (`results/figures/basin_barrier.png`). The truth is a real minimum
and a better one, but it sits behind a barrier about 9% high, and a body must be roughly 85%
of the way there before its misfit beats the convex answer's at all. So descent walks away
from it, and a search has to land inside the last 15% of the path to be rewarded — which a
random draw in 1728 dimensions does not, and which even a dense four-parameter waist search
did not.

The searches make the point sharper than the descent does. In the mixed search, candidate 2
scored **Dice 0.752 at χ 2.349** while the winner scored **Dice 0.696 at χ 2.092**; in the
waist search, candidate 148 scored **Dice 0.747 at χ 2.475**. Better shapes are easy to find.
They have worse misfits. Selecting on misfit therefore selects *against* shape quality, and
that is not a tuning problem.

**What would change it.** The residual at the *true* shape is 0.0375 RMS on model 1 and 0.1058
on model 3 (phase-optimised, so not misalignment). The concavity signal -- the difference
between Mithra's curves and its own hull's -- is 0.1580. Signal exceeds error by 1.5x on the
lab curves and 2.3x on the Blender ones, so the information is there; the barrier is what
hides it, and the barrier's height is set by how much model error a wrong body can absorb.
Halve the forward-model error and the barrier shrinks against the signal. That is the one
lever with real upside left, and it is why the Blender curves are worth calibrating against:
they are a render of the true shape by a known camera with no photographic sensor in front of
it, and they exist for all ten models.

**That lever was tried and it is closed.** `scripts/calibrate.py --blender` fits the instrument
to the Blender curves instead of the lab ones -- a render of the true shape by a known camera,
with no photographic sensor in front of it and no A/B mounting mismatch (their pair difference
is exactly zero). It converges far faster than the lab fit (-logL 383 → -10.7, against
1587 → 1.2) and lands at **eta 0.0372, against the lab fit's 0.0378**. The residual at the
true shape moves from 0.0681 to **0.0667** on model 3 and 0.0255 to 0.0256 on model 1: about
two per cent relative, which is nothing. The mismatch is a property of the forward model, not
of which curves it is calibrated against or of anything the calibration can absorb.

| instrument | eta | model 1 @ truth | model 3 @ truth |
|---|---|---|---|
| all three public bodies, lab curves | 0.0710 | 0.0386 | 0.1078 |
| models 1 and 3 only, lab curves | 0.0378 | 0.0375 | 0.1058 |
| models 1 and 3 only, **Blender** curves | 0.0372 | 0.0256 (blender) | 0.0667 (blender) |

**What this means for a submission.** Ship the convex answers. Any misfit-driven refinement of
them has been measured to make the shape worse -- and note that this includes *convex*
refinement: the Track A run that ended at chi 1.109 stayed at convexity 1.000 throughout, so
it was a convex body fitted to the data, and its Dice still fell 0.690 → 0.665. The convex
stage's answer is good because it is a learned prior over plausible convex bodies, not because
it fits the curves; fitting the curves is what breaks it.

## What the retrained flow changed, and what it did not

Retraining on CSF3 H200s with the two silently-dropped shape families restored (real asteroid
models and Thingi10K objects) took the flow from **5.405 to 5.480**. It is the best flow this
project has produced and it is still **0.064 below the convex stage at 5.544**.

The per-model detail is the interesting part, and it is why no per-model rule is shipped:

| model | convex | retrained flow | winner |
|---|---|---|---|
| 1 Vesta, near-convex | 1.972 | 1.947 | convex |
| 2 sawed cube | 1.901 | 1.842 | convex |
| 3 Mithra, contact binary | 1.672 | **1.691** | **flow** |

The flow wins on the one public body whose shape needs concavity and loses on the two that do
not, which is exactly what it was built to do. A rule that picked per model would score 5.564.
There is no such rule: the obvious candidate, how deeply the flow carved, does not separate
the cases -- model 3's answer has convexity 0.978, between model 1's 0.988 and model 2's
0.897, and the flow wins only on model 3. Picking per model on the public scores would be
fitting to truth the secret models do not come with.

## Where the remaining error actually is

Three measurements, none of which is about the concavity machinery:

**The convex stage under-estimates the body.** Fraction of each truth lying OUTSIDE the convex
answer, which a carve can never recover: model 1 2.9%, model 2 **17.9%**, model 3 3.9%.
Carving alone caps the cube at Dice 0.821 while the convex stage already scores 0.911 there,
so on that body the binding error is the convex inversion, not concavity.

**The representation can already fix this, and does not.** `dh`, the flow's correction to the
support function, is 128 samples band-limited to spherical harmonics of degree 5 -- 36
effective numbers over 4096 normals, which looks far too coarse for a cube. Fitted to the gap
between the convex stage's support and the truth's hull support it captures **99.5%, 99.4% and
98.1%** of it on models 1, 2 and 3. The capacity is there and unused.

**So the limit is the objective, not the parametrisation.** The oracle reaches Dice 0.991 on
Mithra from the convex stage's own support; the line probe shows the misfit rising 9% on the
way to the truth; the forward model misses the lab curves by 4-11% RMS at the *true* shape.
A body that fits the data better is not reliably a body that is more like the truth, and no
change to how concavity is represented alters that.

## Two quirks of the released evaluation code

**`twoDmetric.m`'s `theta` is a no-op.** It rotates both meshes about z and then projects onto
the xy plane; a rotation about z followed by a projection onto xy is an in-plane rotation of
one and the same silhouette. Measured across theta = 0…180° on model 3 the score moves by
0.012, which is rasterisation jitter. As released, the projection measure only ever scores the
**top-down** outline, whatever the challenge text says about "unspecified directions".
`scripts/benchmark.py` reports the genuine side-view reading too (`proj_side`) so a recipe
never comes to depend on the degeneracy.

**Both measures are blind to winding, and only one is blind to holes.** The voxel measure
voxelises by subdividing triangles and flood-filling; the projection measure rasterises each
triangle. Neither cares which way a face points — so the inverted-winding meshes that shipped
for models 2, 4 and 7 would *not* have scored zero. But `.fill()` leaks through an open
surface, so a non-watertight submission is the real hazard. `scripts/check_submission.py`
tests for both.

## Cross-checks

- `hac26/scoring/voxel.py`'s parity-scan Dice tracks the organisers' voxeliser closely and
  reads slightly low (m1 0.9800 vs 0.9786, m2 0.8975 vs 0.9106, m3 0.6914 vs 0.7147). It is
  seconds rather than a minute per model, so it is the right fast proxy inside a loop; quote
  the official number in this table.
- Posing with `centre_xy=True` instead of `False` changes the official voxel score by 0.001
  (m2 0.9106 → 0.9097, m3 0.7147 → 0.7146). The `centre_xy=False` reasoning is right, but the
  scoring consequence is negligible.
- The projection measure centres each mesh on its **vertex mean**, which depends on the
  triangulation. It costs the truth's own convex hull 0.075 on model 2 (0.9165 against the
  convex reconstruction's 0.9905) purely because a 16-face hull and a 2.2M-face truth have
  different vertex means. Submit uniformly and reasonably densely triangulated meshes; ours
  (850–28k faces) are comfortably inside the regime where this does not bite.
- The convex stage reproduces exactly from `models/lpd_convex.pt` on CPU: 6 s per model, Dice
  ≥ 0.999995 against the committed STLs, no GPU and no nvdiffrast.

## Data

`dataset/raw` refreshed 2026-09-08 from the challenge Dropbox. Model 1's four lightcurve files
had been superseded by the 25 August re-release and the local copies were the May ones; the
fresh download agrees hash-for-hash with the manifest on `origin/brendan/fixes`. The
`Evaluation_measures/` folder (released 27 August) had never been fetched at all.
`scripts/check_data.py` verifies the snapshot; `scripts/fetch_data.py` re-fetches it.

What the model-1 re-release actually changed, measured against the superseded copies kept
alongside them: for the **real** curves it is a **pure per-curve circular shift** — realign
each curve by its own offset and the difference is exactly 0.0000, so not one sample value
moved. The shifts run from -5.1 deg to +1.3 deg with five distinct values across the 28
curves. That is the organisers' curve-matching step being redone, and it matters more than a
5-degree number suggests: the shifts differ *per curve*, and the calibration has one start
phase per body, so no value of psi0 could have absorbed them. The old snapshot therefore fed
the calibration a misalignment it could only account for as forward-model error, which is
part of what the fitted eta was absorbing.

The **Blender** curves for model 1 changed in value, not just in phase: 0.0213 RMS before
realignment and 0.0196 after, so realigning explains almost none of it. Those were re-rendered.

**The calibration in `models/` was fitted against the superseded curves, and does not load
anyway** — its keys predate a refactor of `Instrument`, so `Instrument.load` raises. Nothing
that touches the exact forward model runs from a clean checkout until `scripts/calibrate.py`
is rerun.
