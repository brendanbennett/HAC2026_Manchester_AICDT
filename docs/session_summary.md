# Session summary: what was measured, and how to detect training-set bias without ground truth

Two companion documents hold the detail and are not repeated here:

- `docs/gate_audit.md` — the sampling audit, the `_repair` bug, the out-of-family test, full
  tables and the harness invocations.
- `docs/integration_fanyi.md` — what was taken from the `hac-2026-fanyi` tree, what was
  rejected, and why.

This file is the synthesis: the headline numbers, and the part that has no home in either of
those two — a strategy for detecting and mitigating training-set bias on the seven secret
models, where by construction there is no ground truth to score against.

---

## 1. Headline results

### Sampling audit (n = 590, res = 96)

**The base-kind distribution is clean.** Chi-square 7.68, p = 0.47 against `base_weights`; one
outright failure in 590 draws. `prism` sits at 0.098 realised against 0.100 nominal, so the
pathology in HANDOFF §4 (0.12 -> 0.037) is resolved. Because `base_kind` is now drawn once
outside the retry loop, bias in this channel is structurally impossible except through
failure, and the failure rate is 1/590.

**The modifier distribution is not.** `n_modifiers` and each modifier kind are still drawn
INSIDE the retry loop. Chi-square on 2294 slots: 44.1, p = 3.2e-06, with `pitted` at 0.64 of
its declared weight (z = -5.50). `write_report` prints `mod_weights` as though honoured.

### Root cause: a connectivity-pairing bug in `_repair`

Forcing one modifier at a time and logging every attempt: `pitted` was rejected on **71% of
attempts, every one for `multi-component`**, against 3.8% for `craters` and `scallops`.

`_repair` already keeps only the largest solid component, so this should have been impossible.
The parasitic surfaces turned out to be sealed voxel-scale cavities — same volume to four
digits across independent bodies — created because `_repair` labelled solid at 6-connectivity
and background at 26. That is the correct complementary pair for digital topology, but
**marching cubes does not implement that pairing**: a background pocket joined to the outside
only corner-wise is judged "not a void", left alone, and then sealed by the trilinear
interpolant into a genuine interior cavity. `pitted` subtracts 40-90 small spheres and
manufactures these in quantity; `craters` subtracts 2-9 and almost never does.

Labelling background at 6-connectivity fixes all five reproductions at a cost of at most 0.34%
extra voxels filled, and drops `pitted` from 3.43 attempts/body to 1.08.

**The fix is not distribution-neutral — rebuild the corpus, do not patch it.** Median convexity
change over 45 matched seeds is exactly 0.00000, but one body moved by 0.48 and the median
over those 45 shifted 0.680 -> 0.732, because filling cavities removes concave volume.

A strategic note: the pre-fix gate was discarding bodies for a defect with **no observational
consequence** — a lightcurve cannot see an interior void — while `frac convexity < 0.30` sits
at 0.007. It was being selective in a direction that bought nothing and cost deep-neck
coverage.

### Out-of-family representation test

Four meshes built from their own analytic fields, sharing no primitive or modifier with the
library. All four entered unmodified (zero modifiers, attempt 0).

| body | group | truth convexity | core-only Dice | fitted Dice | SDF MSE |
|---|---|---|---|---|---|
| ctrl0-ctrl3 | in-family | 0.544-0.917 | 0.699-0.955 | 0.960-0.997 | 5.6e-05 - 2.1e-04 |
| steps | out | 0.858 | 0.916 | 0.983 | 8.1e-05 |
| cup | out | 0.416 | 0.579 | 0.956 | 1.8e-04 |
| jack | out | 0.163 | 0.278 | **0.946** | 1.1e-03 |
| trefoil | out | 0.251 | 0.400 | 0.946 | 8.5e-04 |

The representation reaches well outside the family it was fitted on: `jack` has a hull 6.1x
its own volume and still decodes to Dice 0.946 at convexity 0.159 against a truth of 0.163.
Mean fitted Dice 0.981 in-family against 0.958 out-of-family.

**Read as a ceiling, not as pipeline performance.** `g` was solved in closed form — the
residual is linear in `g` because Delta is a fixed basis — so the 3000 Adam steps
`fit_shapes.py` runs cannot beat it, and decoding used marching cubes rather than FlexiCubes.

### From the fanyi integration

`results/lpd/` on this branch still carries the cylinder-radius bug (code fixed, artefacts
predate it): model 2 at 0.597 of published R, model 10 at 0.232. Dice = 2s^2/(1+s^2) caps
those at 0.52 and 0.10 regardless of shape quality. Repairing it moved the summed voxel score
on three public models from 2.0468 to 2.5909 in the fanyi tree.

And the most strategically interesting number in that archive: **model 3's convex
reconstruction scores 0.692 while the true model 3's own convex hull scores 0.852 against
it** — roughly 0.16 of headroom before any concavity modelling at all.

---

## 2. Detecting and mitigating training-set bias on the secret models

### First, "training-set bias" is three separate things

Conflating them makes the problem unmeasurable.

1. **Representation bias** — can the field (convex core + Gaussian lattice) express this shape
   at all?
2. **Corpus-coverage bias** — has the flow's prior ever seen anything like it, even if the
   field could hold it?
3. **Data consistency** — does the reconstructed shape explain the 56 measured curves?

The out-of-family test established that (1) and (2) are **separable**: `jack` fits to Dice
0.946, so the field holds it comfortably, while nothing remotely like `jack` was ever in the
corpus. Representational reach does not substitute for coverage, and a diagnostic that only
measures one will miss the other.

### Four signals available without ground truth

**(a) SDF fit residual — detects representation bias.** This was the cleanest result of the
session. Dice separated the two groups by 2.3 points; the residual separated them by an
**order of magnitude** (in-family median 9.4e-05 against out-of-family 5.1e-04, with `jack` and
`trefoil` cleanly outside the control range). It needs no truth, and it is already computed
inside `BatchedFit.loss` and discarded in the mean over bodies. Persist it per body alongside
the corpus; at reconstruction time compare each decoded body's residual against that
distribution.

Two limits, stated plainly: the ranking above came from the closed-form fit and may compress
or reorder under Adam, and it catches representation bias only.

**(b) Code-space OOD distance — detects coverage bias.** This is the signal that speaks
directly to the question, and the architecture gives it away for free: `g` is an **explicit**
1728-dimensional code, with no encoder and no latent vector. So a Mahalanobis or kNN distance
from a reconstructed code to the corpus code distribution is computable directly. The corpus
codes have to be stored anyway, so the marginal cost is near zero. This is the only statistic
that answers "has the flow seen this kind of thing".

**(c) Curve-space data residual — the only signal anchored in real measurement.** Compare the
reconstruction's predicted curves against the 56 measured ones, weighted by the measured
covariance from `covariance.py`. It is **asymmetric and must be used as such**: a high residual
proves the shape is wrong; a low residual does not prove it is right. This codebase contains
its own counterexample — a slotted cube's shadow-free curves can be reproduced *exactly* by a
fatter box. Use it as a falsifier, never as a confirmer.

**(d) Posterior spread — a free ensemble.** `reconstruct_lpd` already draws several samples via
6-step Euler and takes the medoid. The spread of the draws about the medoid is the flow's own
uncertainty. It is already computed; it only needs persisting.

### The step that makes these usable: leave-one-archetype-out calibration

All four signals above are **relative**. Nobody knows what "residual 5e-04" costs in Dice. But
the condition the secret models are in can be manufactured synthetically, where truth exists:

> Build a corpus with one base kind held out — say `arch`. Train. Reconstruct synthetic bodies
> of that held-out kind, measure how much Dice drops, and record the values of (a), (b), (c)
> and (d). Repeat for each of the nine kinds. The result is a calibration curve mapping **OOD
> statistic -> expected Dice loss**.

Then read the secret models' statistics off that curve. This is not guesswork: it is a mapping
learned on a structurally identical problem where the answer is known. It is, in my view, the
only way to genuinely work around the absence of truth rather than talk around it. The
out-of-family test in `docs/gate_audit.md` is a hand-built single point of exactly this shape.

### The ground truth you *do* have about the secret models — spend it first

The organisers published the bounding-cylinder radii: **R = 0.67 to 3.95 against a fixed
z-height of 2**. That is a directly observed aspect ratio for every secret model, and model 10
is roughly 4x wider than tall.

Measured on the corpus in this session, median PCA aspect ratios are s2/s1 = 0.788 and
s3/s1 = 0.585 — near-isotropic, and that regime is **never covered**.

This is the least excusable gap, because it is not a hidden bias at all. Rather than training
one prior for all R, condition on R (or resample/reweight the corpus per model). That converts
an unknown bias into a known covariate.

### Mitigations, ordered by payoff

1. **Fix the `_repair` gate before reasoning about bias at all.** It was discarding bodies for
   an invisible defect while the deep-neck tail sat at 0.7%. Negative cost.
2. **Rebuild the corpus conditioned on the published R** (previous section).
3. **Test-time adaptation.** `map_gauss_newton.py` already exists. Refining each model against
   its own curve residual is the most fundamental defence against coverage bias, because it
   shifts weight from "what training saw" to "what this object's data says".
4. **Ensemble disagreement across deliberately different corpora.** Train two or three
   variants with different biases; disagreement between them is itself a bias indicator and
   needs no truth.
5. **Treat convex-vs-flow disagreement as a red flag.** On model 3 — the only non-convex model
   that can be checked — the flow is currently still *worse* than the convex solver (0.6836 vs
   0.6922). Where the two disagree and the flow is more adventurous, be suspicious.
6. **Keep `eval_gate.py`'s symmetry test.** A flow that has learned concavity will
   **hallucinate** it on the near-convex models, and the voxel measure is a symmetric
   difference, so invented concavity is punished exactly as hard as missed concavity. Model 3
   improving is not sufficient evidence; models 1 and 2 holding is the other half.

### An honest ordering

On the numbers measured here, **bias is not the biggest lever**. The width bug is worth +0.54
from one multiplication. Model 3's convex headroom is worth roughly 0.16 before any concavity
modelling at all. Together those exceed any visible return from the corpus work. The bias
diagnostics are worth building — they are durable and reusable — but they should be sequenced
after those two.

---

## 3. What the codebase looks like now

**The problem.** Recover 3D shape from 56 one-dimensional lightcurves (28 camera geometries x
2 channels: summed intensity and lit-pixel count) of a 3D-printed asteroid model on a
turntable. Scored on voxel overlap plus a 2D projection boundary distance, summed over 7
secret models, max 14.

**The architecture is "convex core plus non-convex correction"**, because the analytic convex
operator `convex_egi` is exact for convex bodies, has a closed-form adjoint and is the only
forward model fast enough for a linear solver — but it is blind to concavity.

**Shape representation** (`hac26/field.py`, the core of everything):

    f(y) = max_j (n_j . y - h_j) + Delta(y)
    Delta(y) = sum_k g_k exp(-||(y - p_k)/sigma||^2 / 2)

4096 spherical-design normals give the convex core as an intersection of half-spaces; Delta is
a signed correction on a fixed 12^3 lattice (sigma = 0.165); **`g` (1728 amplitudes) IS the
code** — no latent vector, no decoder. `h` carries a low-order spherical-harmonic correction
(degree <= 5, 128 dims) because a non-convex body's support function is itself biased. Total
code length 1856.

**Pipeline.** Shape-library generation (`shape_library.py` — level sets plus a convexity gate)
-> code fitting (`fit_shapes.py`, one 1856-dim code per body) -> conditional flow training
(`train_lpd.py`, conditioned on curve spectra) -> reconstruction (`reconstruct_lpd.py`, 6-step
Euler, several draws, medoid) -> `fit_to_cylinder` back to the published radius -> STL.

**Main modules.** `hac26/forward/` interchangeable forward models (`convex_egi`,
`polytope_raycast`, `sdf_surface`, `sdf_volumetric`, `mesh/` which needs a GPU, and
`learned_surrogate`); `hac26/solvers/` (`lpd_convex` unrolled primal-dual, `lpd_flow`
non-convex correction, `minkowski`, `map_gauss_newton`); `hac26/scoring/` (`voxel.py`,
`side_view.py`, the two challenge measures); `hac26/submission.py` validating pose compliance.

**Added to this version in this session:**

- `hac26/shape_library.py::_repair` connectivity fix — the only tracked file modified, one
  functional line.
- `scripts/audit_realised_distribution.py`, `scripts/audit_modifier_gate.py` — sampling and
  per-modifier gate audits, both resumable.
- `scripts/oof_shapes.py`, `oof_ingest.py`, `oof_fit_decode.py` — the out-of-family test, pure
  numpy, no torch required.
- `scripts/eval_gate.py` (from fanyi) — acceptance gate returning a verdict and a non-zero
  exit rather than a number; contains a z-column ray-stabbing occupancy (2.3 s -> 0.03 s).
- `scripts/fix_pose.py` and `results/lpd_fitted/` (from fanyi) — repairs the radius bug above.
- `tools/render.py`, `tools/fig_truth_vs_recon.py` — visual QA, previously absent entirely.
- `docs/gate_audit.md`, `docs/integration_fanyi.md`, this file.
- `pyproject.toml` subpackage list (`pip install -e` was omitting `hac26.forward`, `solvers`,
  `scoring` and FlexiCubes) and `.gitignore` allowing `runs/gate_*.json` to be tracked.

**Two memory exposures found while measuring.** `_parity_occupancy` allocates about 1.1 GB per
chunk at res = 96 regardless of mesh size, because the chunk is fixed at 3000 triangles.
`fit_shapes.sample_arrays` calls `trimesh.nearest.signed_distance` unchunked — measured 1.5 GB
for 2000 points against a 58k-face body, so its 9000 points need roughly 7 GB on that body,
and 40-60k-face library bodies are common.

**Not done.** The audit reached 590 of 1000; re-run it after the `_repair` fix and check
whether `pitted`'s z of -5.50 collapses — that is the direct test of whether p = 3.2e-06 had
one cause or several. `bite` (+2.27) and `scallops` (+2.24) were never investigated. Nothing
here was run end to end through `fit_shapes` / `train_lpd` with torch.
