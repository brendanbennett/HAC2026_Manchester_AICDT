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
**The calibration in `models/` was fitted against the superseded model-1 curves.**
