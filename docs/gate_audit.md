# Audit: realised sampling distribution, and a `_repair` bug the gate was hiding

Every number here was measured on this tree. Harnesses are committed alongside so each can be
re-run; all three are resumable, because the runs are hours on one core.

    scripts/audit_realised_distribution.py    realised base/modifier distribution
    scripts/audit_modifier_gate.py           per-modifier rejection rate, logging EVERY attempt
    scripts/oof_shapes.py                    four deliberately out-of-family meshes
    scripts/oof_ingest.py                    body_from_mesh ingestion, one shape per call
    scripts/oof_fit_decode.py                fit + decode + Dice, in numpy (see its header)

---

## 1. The base-kind distribution is clean

n = 590 drawn bodies, `LibrarySpec()` defaults, res = 96. **One** failure (a `contact_binary`
that exhausted all 12 attempts with "empty body: the field is positive everywhere").

Chi-square across the nine kinds: **7.68, p = 0.47.** The realised distribution is
indistinguishable from `base_weights`.

| kind | expected | observed | z |
|---|---|---|---|
| star_sh | 100.3 | 106 | +0.62 |
| contact_binary | 94.4 | 96 | +0.18 |
| lobes | 88.5 | 95 | +0.75 |
| rubble | 70.8 | 71 | +0.03 |
| polytope | 59.0 | 46 | -1.78 |
| prism | 59.0 | 58 | -0.14 |
| ellipsoid | 47.2 | 53 | +0.88 |
| slab | 41.3 | 31 | -1.66 |
| arch | 29.5 | 34 | +0.85 |

`prism` sits at 0.098 realised against 0.100 nominal, so the pathology recorded in HANDOFF §4
(prism 0.12 -> 0.037) is resolved. Drawing `base_kind` once outside the retry loop makes any
further bias in this channel impossible EXCEPT through outright failure, and the measured
failure rate is 1/590.

**Unrelated but load-bearing:** `sample_body` raises `RuntimeError` when it exhausts
`max_attempts`, and `scripts/build_shape_library.py::_worker` does not catch it -- the
exception propagates out of `imap_unordered` and kills the pool. At 1/590, a 5000-body build
hits one with probability ~0.9998. Catch it, log the base kind, redraw on a derived seed; the
failure-rate telemetry comes free.

## 2. The modifier distribution is NOT clean

`n_modifiers` and each `mk = _draw(spec.mod_weights, rng)` are still drawn INSIDE the retry
loop, which is the shape of the bug that was fixed for `base_kind`. Chi-square on 2294
modifier slots: **44.1, p = 3.2e-06**, and `write_report` prints `mod_weights` as though
honoured.

| modifier | realised / nominal | z |
|---|---|---|
| pitted | **0.64** | **-5.50** |
| bite | 1.13 | +2.27 |
| scallops | 1.17 | +2.24 |
| tunnel | 1.25 | +1.63 |
| the other seven | 0.92 - 1.07 | within +-1.2 |

The `pitted` deficit is already present among bodies accepted on attempt 0 (0.055 against a
nominal 0.100) and is HIGHER among retried bodies (0.072). That rules out modifier
intensification -- `strength` only rises on a retry -- and points at per-attempt rejection.

## 3. Per-attempt rejection, measured directly

`mod_weights` forced to one kind, base distribution left alone, every attempt logged
(`sample_body` returns only the survivor, so its loop is replicated rather than called):

| modifier | attempts/body | rejections | reason |
|---|---|---|---|
| pitted | **3.43** | 17/24 (71%) | **all `multi-component`** |
| craters | 1.04 | 1/26 (3.8%) | convexity gate |
| scallops | 1.04 | 1/26 (3.8%) | convexity gate |

Not the convexity gate. Not manifoldness. Every `pitted` rejection is `n_components != 1`.

## 4. Which should have been impossible, and why it was not

`_repair` already keeps only the largest solid component, so a second SURFACE component should
not survive it. Characterising one:

    body 1  base=slab  repair={'n_solid_components': 1, 'n_voids_filled': 0}
            surfaces=2   faces/volume: [(20052, 1.512853), (216, 0.002264)]

One solid component, zero voids filled, two surfaces. The parasitic surface carries volume
0.002264 -- and the same figure recurs across four independent bodies (0.002264, 0.002266,
0.002268, 0.002275). A repeated fixed volume is a sealed voxel-scale cavity, not a detached
fragment: a detached fragment would have been discarded by the solid-component step, and its
volume would vary.

The cause is the connectivity pairing. `_repair` labelled solid at 6-connectivity and
background at 26, which is the correct complementary pair for digital topology -- but marching
cubes does not implement that pairing. A background pocket joined to the outside only
corner-wise is judged "not a void", left alone, and then SEALED by the trilinear interpolant
into a genuine interior cavity, i.e. a second closed surface. `pitted` subtracts 40-90 small
spheres near the surface and manufactures these pockets in quantity; `craters` subtracts 2-9
and almost never does.

Labelling the background at 6-connectivity instead resolves all five reproductions:

| body | 26-conn (before) | 6-conn (after) | extra voxels filled |
|---|---|---|---|
| 0 | 2 surfaces | 1 | 67 |
| 1 | 2 | 1 | 130 |
| 3 | 2 | 1 | 56 |
| 4 | 4 | 1 | 572 |
| 5 | 5 | 1 | 3015 |

Worst case 3015 voxels of 884,736 -- 0.34%.

## 5. After the fix

    pitted (before)   bodies   7  attempts 24  att/body 3.43  {ACCEPTED: 7, multi-component: 17}
    pitted (after)    bodies  25  attempts 27  att/body 1.08  {ACCEPTED: 25, convexity gate: 2}

`tests/test_shape_library.py`: 25 passed, 3 deselected.

On the natural distribution over the same 45 seeds as §1: base kind identical in **45/45**,
retry rate halves (0.222 -> 0.111), median convexity change exactly 0.00000.

**The fix is not distribution-neutral and the corpus must be REBUILT, not patched.** One body
of the 45 moved by 0.48 in convexity -- it had a large sealed cavity that is now filled -- and
the convexity median over those 45 shifted 0.680 -> 0.732, because filling cavities removes
concave volume. Both are correct: those cavities were artefacts the extractor invented, and an
interior void is invisible to a lightcurve either way. But codes fitted against the old corpus
were fitted against bodies containing them.

Worth noting against HANDOFF §8.5: the pre-fix gate was discarding bodies for a defect with no
observational consequence, while `frac convexity < 0.30` sits at 0.007. It was being selective
in a direction that costs deep-neck coverage without buying anything.

## 6. Out-of-family representation test

Four meshes built from their own analytic fields, sharing no primitive or modifier with the
library (`scripts/oof_shapes.py`): `jack` (six slender arms), `cup` (deep cavity behind a
narrow mouth), `trefoil` (knotted tube, no bulk anywhere), `steps` (rectilinear staircase).
All four passed `body_from_mesh` with zero modifiers at attempt 0, and ingestion preserved
their convexity (jack 0.165 -> 0.163), so they entered unmodified.

| body | group | truth convexity | core-only Dice | fitted Dice | decoded convexity | SDF MSE |
|---|---|---|---|---|---|---|
| ctrl0 | in | 0.544 | 0.699 | 0.978 | 0.573 | 2.1e-04 |
| ctrl1 | in | 0.641 | 0.781 | 0.960 | 0.639 | 5.9e-05 |
| ctrl2 | in | 0.917 | 0.955 | 0.997 | 0.932 | 1.3e-04 |
| ctrl3 | in | 0.827 | 0.903 | 0.987 | 0.830 | 5.6e-05 |
| steps | **out** | 0.858 | 0.916 | 0.983 | 0.864 | 8.1e-05 |
| cup | **out** | 0.416 | 0.579 | 0.956 | 0.435 | 1.8e-04 |
| jack | **out** | 0.163 | 0.278 | **0.946** | 0.159 | 1.1e-03 |
| trefoil | **out** | 0.251 | 0.400 | 0.946 | 0.253 | 8.5e-04 |

The representation reaches well outside the family it was fitted on. `jack` has a hull 6.1x
its own volume and still decodes to Dice 0.946 at convexity 0.159 against a truth of 0.163.
Mean fitted Dice 0.981 in-family against 0.958 out-of-family.

**Read this as a CEILING, not as pipeline performance.** `scripts/oof_fit_decode.py` solves g
in closed form -- the residual is linear in g because Delta is a fixed basis -- so the 3000
Adam steps `fit_shapes.py` runs from g = 0 cannot beat it. Any shortfall is the
representation's, and the real pipeline can only do worse. Decoding also uses marching cubes
rather than FlexiCubes (torch was not installable in the measuring environment), so do not
quote these against FlexiCubes numbers to three decimals.

**The useful column for the secret models is SDF MSE, not Dice.** Dice separates the two
groups by 2.3 points; the residual separates them by an order of magnitude -- in-family median
9.4e-05 against out-of-family 5.1e-04, with `jack` and `trefoil` cleanly outside the control
range. Dice needs ground truth. The residual does not, and it is already computed inside
`BatchedFit.loss` and discarded in the mean over bodies. Persisting it per body alongside the
corpus gives a screening statistic that survives the absence of truth.

Two limits on that. The ranking above came from the closed-form fit and may compress or
reorder under Adam. And it measures representational REACH, not corpus coverage: `jack` fits
well, but nothing resembling it was ever in the training distribution, so the flow's prior
over g may refuse to generate it even though the field can hold it. Those are separate
failures and the residual only catches the first.

## 7. Two memory exposures found while measuring

Neither is a correctness bug; both bite on a big library or a parallel build.

- `_parity_occupancy` allocates five `(res^2, 3000)` float64 arrays per chunk -- about 1.1 GB
  at res = 96 -- regardless of mesh size, because the chunk is fixed at 3000 triangles.
- `fit_shapes.sample_arrays` calls `trimesh.nearest.signed_distance` unchunked. Measured
  1.5 GB RSS for 2000 points against a 58k-face body, so the 9000 points it asks for need
  roughly 7 GB on that body. Library bodies at 40-60k faces are common (two of the four
  controls above). With `--workers 16` this is a live exposure on the remote build.

## 8. What has not been done

- The audit reached **590 of 1000** bodies. It is resumable; re-run it AFTER the `_repair` fix
  and check whether the `pitted` z of -5.50 collapses. That is the direct test of whether the
  p = 3.2e-06 had one cause or several.
- `bite` (+2.27) and `scallops` (+2.24) were not investigated. They may be the same mechanism
  with the opposite sign, or multiple-comparison noise across eleven modifiers.
- Nothing here was run end to end through `fit_shapes` / `train_lpd` with torch.
