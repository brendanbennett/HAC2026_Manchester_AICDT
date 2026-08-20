# Merge notes: `Jacks_shapes-fixed` (2026-08-19) into `main` (2026-08-13)

This tree is `hac-2026-main` with the four code changes from `hac-2026-Jacks_shapes-fixed`
applied, plus five small repo-hygiene fixes made necessary by them. Nothing from `main`
was removed or rewritten.

`CHANGES.md` (from the fixed archive) states what the changes are and why. This file
records only what the merge itself did, and what was and was not verified.

## Why this merge is low-risk

The fixed archive is not a divergent branch, it is `main` plus additions. A file-by-file
diff of the two archives:

    identical                 108 of 110 files
    modified                  scripts/reconstruct_lpd.py, scripts/train_surrogate.py
    new                       hac26/shapes_nonconvex.py, scripts/fix_pose.py,
                              scripts/eval_gate.py, CHANGES.md,
                              results/lpd_fitted/, runs/gate_*.json
    absent from the archive   .gitignore, dataset/   (packaging omission, kept from main)

No file in `hac26/` other than the new `shapes_nonconvex.py` is touched, so every forward
model, solver, scoring routine and checkpoint behaves exactly as before.

## Blast radius of the two modified files

**`scripts/reconstruct_lpd.py`** — one block added before `export_stl`, guarded by
`--no-fit-cylinder`, plus a sibling import of `fix_pose.min_enclosing_radius`. That import
is safe: the script already puts its own directory on `sys.path` (it does the same thing
for `train_lpd`). The convex path (`reconstruct.py`, `eval_exact.py`) is untouched, so
`results/convex/` — the submitted set — is unaffected.

**`scripts/train_surrogate.py`** — `shapes()` becomes a wrapper over
`shapes_nonconvex.sample_corpus`. This is the one change with real reach: `fit_shapes.py`
imports `shapes` from here, so the token library and, through it, `train_lpd.py` all draw
from the new corpus. It changes nothing until something is retrained, and the surrogate
feature cache key was bumped `v7_` -> `v9_` so a stale `/tmp` cache cannot silently mix
corpora. To get the old behaviour back, revert this one file; `sample_corpus` is not
imported anywhere else.

## Repo-hygiene fixes made during the merge

These were not in either archive. Each exists because merging exposed it.

1. **`.gitignore`: `runs/` -> `runs/*` + `!runs/gate_*.json`.** `main` ignores `runs/`
   wholesale, so committing this tree as-is would have silently dropped the gate
   baselines that `CHANGES.md` tells you to compare against. Git cannot re-include a file
   whose parent directory is excluded, hence `runs/*` rather than `runs/`.

2. **`requirements.txt` / `pyproject.toml`: `trimesh` and `rtree` declared.** Both were
   already required by `main` (`hac26/recon.py`, `hac26/solvers/output.py`) and never
   listed. `rtree` matters more than it looks: trimesh's pure-python ray engine needs it,
   and without it every `mesh.ray.intersects_location` raises — which
   `sample_corpus` catches, so it returns an **empty corpus** instead of an error.
   Reproduced here: `python -m hac26.shapes_nonconvex --n 8` printed
   `WARNING: 0/8 after 112 attempts` with no traceback until `rtree` was installed.
   Worth knowing before a training run appears to succeed on nothing.

3. **`scikit-image` declared** — `shapes_nonconvex._mesh_from` uses marching cubes.

4. **`scipy>=1.10` -> `scipy>=1.15`.** `shapes_nonconvex._sh_basis` calls
   `scipy.special.sph_harm_y`, which replaced `sph_harm` in SciPy 1.15 and does not exist
   in 1.10.

5. **`pyproject.toml` package list.** It read `packages = ["hac26"]`, which installs only
   the top-level module — `pip install -e .` left `hac26.forward`, `hac26.solvers`,
   `hac26.scoring` and the vendored FlexiCubes out. Pre-existing in `main`; listed the
   subpackages explicitly. (`hac26.vendor` has no `__init__.py`, so only
   `hac26.vendor.flexicubes` is listed.)

`README.md` gained the `fix_pose.py` and `eval_gate.py` invocations and two lines in the
structure block. `runs/gate_test.json` (a one-model scratch file) was dropped;
`scripts/.DS_Store` was not copied.

## Verified here

* `py_compile` clean across all 116 Python files.
* The numpy-only import surface loads: `hac26.shapes`, `recon`, `geometry`, `data_io`,
  `scoring.voxel`, `scoring.side_view`, `shapes_nonconvex`.
* `eval_gate.py --help`, `fix_pose.py --help` both parse.
* `fix_pose.py` on `results/lpd/Asteroid02.stl` reproduces the shipped
  `results/lpd_fitted/Asteroid02.stl` **bit for bit** (max |vertex difference| = 0.0,
  `R 0.840 -> 1.406`, xy x 1.674). So `results/lpd_fitted/` is exactly what the committed
  script produces, not a hand-edited artefact.
* `python -m hac26.shapes_nonconvex --n 8` fills every stratum as designed:

      near_convex      3   D_rms 0.000              R 0.90-1.40
      nonconvex_bin0   1   D_rms 0.061  D_lo 0.76   R 1.52
      nonconvex_bin1   1   D_rms 0.162  D_lo 0.72   R 1.22
      nonconvex_bin2   1   D_rms 0.210  D_lo 0.83   R 0.77
      nonconvex_bin3   1   D_rms 0.276  D_lo 0.84   R 1.55
      nonconvex_bin4   1   D_rms 0.337  D_lo 0.74   R 0.98

## Not verified here — needs your machine

* Anything touching torch (not installed in this sandbox): `reconstruct_lpd.py`,
  `train_lpd.py`, `fit_shapes.py`, the checkpoints in `models/`.
* `eval_gate.py` end to end: it needs the organisers' public ground-truth STLs, which are
  in neither archive. The numbers in `runs/gate_*.json` are Jack's, not re-measured here.
* `pytest tests/` — pytest is not installed here, and several tests read `dataset/raw`.

## Two things to decide, not merged

* **Two different xy-centring conventions.** `fix_pose.repose` and the new block in
  `reconstruct_lpd.py` centre on the **minimum enclosing circle** of the xy projection;
  `eval_gate._center_xy` centres on the **volume centroid**, and its own docstring cites
  the challenge wording ("centroid on the axis") as the reason. The exporter and the
  compliance checker therefore do not agree, and on an unevenly tessellated mesh they can
  differ by a few percent of R. Left as found — picking one is a judgement about what the
  organisers will measure, not a merge decision.
* **Three different default data paths.** `reconstruct_lpd.py`, `fit_covariance.py`,
  `eval_gate.py` and `tests/` default to `dataset/raw`; `reconstruct.py`,
  `eval_exact.py` and `scoring/side_view.py` default to `../data/raw`;
  `calibrate.py` hardcodes `data/raw/...`; and `dataset/MANIFEST.sha256` lists its 198
  files under `data/raw/`, so `sha256sum -c` only passes from a directory where that
  path resolves. Pre-existing in `main`, unrelated to this merge, and worth one commit.

## Suggested first commit sequence

    git checkout -b cylinder-fix
    # 1. the bug fix alone: scripts/reconstruct_lpd.py, scripts/fix_pose.py,
    #    results/lpd_fitted/, runs/gate_*.json, .gitignore, CHANGES.md
    # 2. the gate: scripts/eval_gate.py
    # 3. the corpus bet: hac26/shapes_nonconvex.py, scripts/train_surrogate.py
    # 4. hygiene: requirements.txt, pyproject.toml, README.md

Commit 1 is worth +0.54 summed voxel on the three public models and is independent of
everything else; commit 3 is an untrained hypothesis. Keeping them apart means the corpus
can be reverted without giving the radius fix back.
