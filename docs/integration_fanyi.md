# Integrating `hac-2026-fanyi` into this branch

`hac-2026-fanyi` is not a descendant of this tree. It forks from `main` (2026-08-13) plus
the `Jacks_shapes-fixed` changes, and predates the work in `docs/HANDOFF.md` — it still has
`TokenField`, no `GaussianLattice`, no `hac26/shape_library.py`, no `design4096.npy`. So most
of it is superseded. What follows is what was taken, what was not, and why.

The four modules its useful scripts import — `hac26/recon.py`, `hac26/scoring/side_view.py`,
`hac26/scoring/voxel.py`, `hac26/shapes.py` — are **byte-identical** across the two trees, so
the taken files are drop-ins, not ports.

---

## Taken

### `scripts/eval_gate.py`

An acceptance gate on the three public models: challenge voxel Dice, projection-boundary
ASSD, pose compliance, and a **verdict** with a non-zero exit rather than a number. Its
argument is worth restating because it applies to everything on this branch: the failure mode
of a corpus that has learned concavity is symmetric to the one it fixes — the flow will
HALLUCINATE concavity on the near-convex models, and the voxel measure is a symmetric
difference, so an invented concavity is punished exactly as hard as a missed one. Model 3
improving is not sufficient evidence. Models 1 and 2 holding is the other half.

This branch had no such harness. It is torch-free and imports no training code, so it runs on
the machine holding the dataset without a GPU.

Two pieces inside it are independently valuable:

- **`occupancy()` by z-column ray stabbing.** `hac26/scoring/voxel.py` still voxelises with
  `mesh_to_sdf`, which runs `mesh.contains` at every one of n³ grid points and then two
  distance transforms — 2.1M point-in-mesh queries at n=128, and the distance transforms are
  pure waste since Dice needs occupancy, not distance. Stabbing one ray up each z-column is
  n² queries: 16384 instead of 2.1M, verified identical voxel count and Dice 1.00000 on model
  3, **2.3 s → 0.03 s**. `voxel.py` was deliberately NOT changed here — it is the submission
  scoring path and the dataset is not present to re-verify against — but it should adopt this
  once someone can run it against the public STLs.
- **`pose_check()` separating `max_axis_dist` from `min_enclosing_R`.** This resolves what
  looks like a contradiction between the two trees. `hac26/conventions.py` says two of the
  three public bodies EXCEED their published R; `scripts/fix_pose.py` says the min-enclosing
  radius over published R is 0.99/0.99/0.98. Both are right: one is measured about the
  origin, the other after re-centring. The distinction matters operationally — a body that
  fits once re-centred is sitting off-axis and a translation fixes it (a warning), while one
  whose min-enclosing radius exceeds R is genuinely too wide (a rejection). Note also that
  `_center_xy` uses the VOLUME centroid, because `rescale_touch_z` centres on the vertex mean
  and on the cube that lands 2% off axis, which reads R = 1.52 against a published 1.42 on a
  body that is actually compliant.

### `scripts/fix_pose.py` and `results/lpd_fitted/`

**This is the item with an immediate measurable effect, and it applies to this branch.**

`scripts/reconstruct_lpd.py` on this branch already calls `fit_to_cylinder(v, R)` — HANDOFF
§3.4 fixed the frame problem by a different route. But **the committed STLs in `results/lpd/`
predate that fix and are still wrong.** Measured here, ratio of achieved radius to published
R:

| model | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| `results/convex/` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `results/lpd/` | 0.900 | **0.597** | 1.030 | **0.654** | 0.777 | 0.918 | 0.773 | 0.782 | 1.011 | **0.232** |

For a body scaled by s in xy and otherwise contained in the truth, Dice = 2s²/(1+s²). At
s = 0.597 the score is capped at 0.52 however good the shape is; at s = 0.232 it is capped at
0.10. Nothing about the corpus or the flow can fix a width error.

`results/lpd_fitted/` (all ten, from the fanyi tree) is exactly this repaired. Provenance
verified rather than trusted: this branch's `results/lpd/` STLs are byte-identical to the
fanyi tree's, and re-running `fix_pose.py` here on models 2 and 10 reproduces the shipped
`lpd_fitted` meshes to **max|diff| = 0.0**.

Measured effect in the fanyi tree, against the public STLs at `--fast` settings:

```
                  model 1   model 2   model 3   summed voxel
lpd (as it was)    0.8799    0.4988    0.6680      2.0468
lpd_fitted         0.9727    0.9346    0.6836      2.5909
convex (submitted) 0.9792    0.9128    0.6922      2.5843
```

**+0.54 summed on three models from one multiplication**, which puts the re-posed LPD set
narrowly ahead of the submitted convex set.

One caveat carried over rather than adopted: `fix_pose.py` scales to **0.99 R** by default,
arguing the published R is an upper bound the bodies nearly touch. That reasoning conflicts
with `hac26/conventions.py`, which treats the published radius as an approximation that two
public bodies exceed. The default was left as fanyi wrote it; the flag is `--cylinder-fill`,
and this is a real open disagreement, not a settled convention.

### `tools/render.py`, `tools/fig_truth_vs_recon.py`

Shaded orthographic renders from the occupancy grid — first occupied voxel along the view
axis gives a depth map, its gradient gives normals. Chunky at the silhouette, but it treats a
6k-face reconstruction and an 800k-face ground truth identically, which is the point of a
comparison figure. This branch has no visual QA at all, and every convexity number in
HANDOFF §5 and `docs/gate_audit.md` would be easier to trust next to a picture.

### `pyproject.toml` — the subpackage list

`packages = ["hac26"]` installs only the top-level module, so `pip install -e ".[torch]"` —
the command in the README — leaves out `hac26.forward`, `hac26.solvers`, `hac26.scoring` and
the vendored FlexiCubes. Pre-existing on both trees; the explicit list is taken.

### `.gitignore` — `runs/` → `runs/*` + `!runs/gate_*.json`

Git cannot re-include a file whose parent directory is excluded, so the wholesale `runs/`
ignore would silently drop the gate baselines that the gate exists to compare against.
`runs/gate_{baseline,convex,lpd,lpd_fitted}.json` come from the fanyi tree and are that
branch's measurements — treat them as its baselines, not this one's.

---

## Not taken

- **`hac26/shapes_nonconvex.py`** — the hull-deficit-stratified corpus. Superseded by
  `hac26/shape_library.py`, which is strictly further along: level sets rather than mesh
  booleans (so one component, no interior void and a closed surface are decidable and
  repairable on the occupancy grid before a triangle exists), a non-convexity GATE rather
  than a hope, and ingestion paths for Thingi10K and DAMIT. Taking it would be a regression.
- **`tools/token_init_test.py`, `tools/token_sweep.py`** — measure the `TokenField`
  zero-init defect. `TokenField` no longer exists; HANDOFF §3.1 records the same finding.
- **`tools/fit_corpus.py`** — written against the old `ImplicitBody` API (shared decoder,
  cross-attention tokens). `scripts/fit_shapes.py` covers this on the current field.
- **`tools/fig_coverage.py`, `tools/make_corpus_sample.py`, `corpus_sample/`** — import
  `shapes_nonconvex`. `fig_coverage.py` is worth re-pointing at `shape_library`'s convexity
  distribution later; it was not adapted here.
- **Their `scripts/reconstruct_lpd.py` and `scripts/train_surrogate.py`** — both superseded
  by larger rewrites on this branch.
- **Dependency declarations** — this branch already declares `trimesh`, `rtree` and
  `scikit-image`. Their `scipy>=1.15` bump exists for `shapes_nonconvex._sh_basis`'s use of
  `sph_harm_y`; nothing here needs it, so `scipy>=1.10` stands.

Worth keeping from their `MERGE_NOTES.md` even though the dependency fix is already in:
without `rtree`, trimesh's pure-python ray engine raises on every
`mesh.ray.intersects_location`, and a corpus sampler that catches exceptions returns an
**empty corpus rather than an error**. `eval_gate.occupancy` uses that same call.

---

## Carried forward as findings, not code

From their `CHANGES.md`, measured on the fanyi tree and not reproduced here:

1. **On model 3 the flow is still slightly worse than the convex solver** (0.6836 vs 0.6922).
   The non-convex correction is not yet paying for itself on the only non-convex model that
   can be checked.
2. **Model 3's convex reconstruction is far from the achievable convex answer.** The true
   model 3's own convex hull scores 0.852 against it; the convex stage delivers 0.692. That
   is ~0.16 of headroom *before any concavity modelling at all* — about the same size as the
   entire remaining gap from the hull to a perfect reconstruction (0.852 → 1.0). If that
   number holds, it is a better target than anything in the corpus work.
3. **Convex models 6 and 7 sat at 0.86 and 0.84 of R** in their measurement while everything
   else was at 0.97–0.99. Measured on this branch's `results/convex/` they are all exactly
   1.000, so either the sets differ or the two measurements centre differently — see the
   `max_axis_dist` vs `min_enclosing_R` distinction above. Resolve before acting on it.
