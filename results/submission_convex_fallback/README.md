# HAC 2026 submission — the seven scored models

`Asteroid04.stl` … `Asteroid10.stl`, in the challenge pose: rotation axis is z, the body
touches z = +1 and z = −1, light at (−∞, 0, 0), pose = frame 0 of the lightcurves. Each is a
single watertight component of positive volume with consistent winding, and each touches its
published bounding cylinder exactly (`scripts/check_submission.py` verifies all of it and
exits non-zero if anything fails).

## What produced them

The convex stage alone: an unrolled learned primal-dual network on the analytic convex
operator (`hac26/solvers/lpd_convex.py`, checkpoint `models/lpd_convex.pt`), with the xy scale
set to the published bounding-cylinder radius.

```
python scripts/reconstruct.py --ckpt models/lpd_convex.pt --model N --fit-cylinder \
    --out results/submission/AsteroidNN.stl
```

Six seconds a model on a CPU. No GPU, no nvdiffrast, no trained flow.

## Why the convex answer, and not a non-convex one

Scored with the organisers' own code (`hac26/scoring/official.py`, `scripts/benchmark.py`) on
the three public models, out of a maximum of 6:

| pipeline | total |
|---|---|
| convex hull of truth — the convex ceiling | 5.775 |
| **this submission** | **5.544** |
| the previous `results/lpd`: convex + trained flow + polish | 5.405 |

The non-convex stage cost 0.14 and gained 0.007 on the one public body whose shape needs
concavity. That is not a tuning failure, and `BENCHMARKS.md` has the measurements: the true
shape sits behind a barrier in the misfit. Walking from the convex answer to an oracle — the
lattice fitted by least squares to the true body, on this same convex support — the misfit
rises 2.147 → 2.336 before falling to 1.898, while Dice climbs 0.690 → 0.991 the whole way. A
body has to be about 85% of the way to the truth before its misfit beats the convex answer's.

So every method that selects on misfit selects against shape quality, and three independent
ones did: gradient descent on the exact misfit reached χ 1.109 at Dice 0.665; a 250-candidate
search over carved bodies found χ 2.092 at Dice 0.696 while a *different* candidate sat at
Dice 0.752 with a worse misfit of 2.349; a 400-candidate dense search over the four-parameter
waist family — the family that contains a contact binary, which model 3 is — produced nothing
that beat convex at all.

This holds inside the convex class too, where the inverse problem is well posed: the descent
run that reached χ 1.109 never left convexity 1.000, and its Dice still fell. The convex
stage's answer is good because it is a learned prior over plausible convex bodies, not because
it fits the curves; fitting the curves is what breaks it.

The barrier's height is set by how much model error a wrong body can absorb, and the forward
model misses the curves at the *true* shape by 0.0375 RMS on model 1 and 0.1058 on model 3.
Calibrating against the Blender curves instead of the lab ones does not move it (0.0667 vs
0.0681 at truth). Halving that error is where the next real gain is, and it is a forward-model
problem rather than a solver one.

## Caveat

Model 3 is the only public body with genuine concavity, so the barrier, the oracle and the
signal-to-error ratio all rest on one body. The conclusion is that carving *on this evidence*
costs score; it is not a proof that no non-convex method can work.
