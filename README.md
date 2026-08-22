# hac26

Shape reconstruction from lightcurves for the
[Helsinki Asteroid Challenge 2026](https://fips.fi/data-challenges/helsinki-asteroid-challenge-2026/).

Ten 3D-printed asteroids were filmed on a turntable from 28 camera geometries. Each frame is
reduced to two numbers, summed intensity and lit-pixel count, giving 56 curves per body.
Reconstructions are scored on voxel overlap with the true shape plus the distance between the
boundary curves of 2D projections. `docs/challenge_info.md` has the rules.

## Install

```
pip install -e ".[torch]"
```

Only the mesh rasteriser needs a GPU. It uses nvdiffrast, which is not on PyPI and compiles
CUDA at install time; `scripts/setup_toolchain.sh` builds it without root. Everything else
runs on CPU.

## Data

Put the challenge data in `dataset/raw/` under the organisers' original directory names and
check it against `dataset/MANIFEST.sha256`. It is not stored in the repo.

## Running the pipeline

Two wrappers do the whole thing:

```
scripts/run_smoke_test.sh        # wiring check: minutes, CPU only, no dataset needed
scripts/run_remote_pipeline.sh   # the real run
```

`run_remote_pipeline.sh` creates and activates a `.venv`, installs anything missing, and runs
the stages in order:

| stage | script | notes |
| --- | --- | --- |
| `library` | `build_shape_library.py` | CPU only |
| `design` | `make_design.py` | GPU if there is one |
| `calibrate` | `calibrate.py` | needs `dataset/raw`; skipped if `models/instrument_calibration.pt` is present |
| `surrogate` | `train_surrogate.py` | needs `dataset/raw` and nvdiffrast |
| `fit` | `fit_shapes.py` | fits shape codes to the library |
| `flow` | `train_lpd.py` | |
| `reconstruct` | `reconstruct_lpd.py` | all ten models |
| `score` | `hac26/scoring/` | the three public models |

Each stage writes a marker under `runs/.done/` and is skipped if it is already there, so a
dropped run is safe to relaunch. Output goes to `logs/<stage>.log`. Every setting is a
variable at the top of the script and can be overridden from the environment:

```
N_BODIES=2000 scripts/run_remote_pipeline.sh
scripts/run_remote_pipeline.sh --force-stage fit    # redo fit and everything after it
```

## Running stages by hand

Build the set of surface normals the shape representation is defined on. Once per size:

```
python scripts/make_design.py --n 4096
```

Fit the instrument to the real curves of the three public models:

```
python scripts/calibrate.py
```

Build a shape library, fit codes to it, train the flow:

```
python scripts/build_shape_library.py --n 5000
python scripts/fit_shapes.py --shapes-dir dataset/generated/shapes --bodies 2000
python scripts/train_lpd.py --bodies 2000 --steps 4000
```

The forward surrogate the flow inverts is trained separately:

```
python scripts/train_surrogate.py --train 256 --held 16 --steps 6000 --workers 8
```

`--workers` parallelises the CPU half of the dataset build; the rendering stays in one
process because it holds the CUDA context. `--phases` has to agree between
`train_surrogate.py`, `train_lpd.py` and `reconstruct_lpd.py`; they share a default and the
surrogate checkpoint records what it was trained at, so a mismatch is reported rather than
silently absorbed.

`fit_shapes.py` and `train_lpd.py` both checkpoint and resume, so `--steps` is a cap and not
a schedule. `train_lpd.py` also early-stops on a held-out split (`--val-bodies`,
`--val-every`, `--patience`).

Reconstruct one model, with the flow or with the convex solver:

```
python scripts/reconstruct_lpd.py --model 4 --out results/lpd/Asteroid04.stl
python scripts/reconstruct.py --ckpt models/lpd_convex.pt --model 4 --out results/convex/Asteroid04.stl
```

Score against a public model:

```
PYTHONPATH=. python hac26/scoring/voxel.py --stl results/lpd/Asteroid01.stl --models 1
PYTHONPATH=. python hac26/scoring/side_view.py --models 1 2 3 --recon-dir results/lpd
```

Check that the flow is using the lightcurves rather than memorising the corpus, by
rerunning its own validation with every curve-carrying channel zeroed:

```
python scripts/ablate_flow.py
```

## Tests

```
pytest
pytest -m "not slow"     # skip the extraction-scale ones
```

## Layout

```
hac26/forward/    forward models, one per file; see forward/__init__.py
hac26/solvers/    lpd_convex, lpd_flow, minkowski, map_gauss_newton, curve_loss, output
hac26/scoring/    voxel, side_view
hac26/            conventions, geometry, field, shapes, calibrate, noise, covariance,
                  shape_library, curves_mesh, library_io, library_metrics
scripts/          entry points
models/           trained convex solver and instrument calibration; models/load.py loads them
results/          reconstructions: convex/ is the submitted set, lpd/ the flow solver's
dataset/          challenge data, not tracked
runs/             training output, not tracked
tests/
```

Forward models are interchangeable and differ in shape parameterisation and physics, from an
analytic convex operator with no shadowing to a full mesh chain with interreflection and a
camera model. `hac26/forward/__init__.py` lists what each one covers and leaves out.

`lpd_convex` is an unrolled primal-dual network on the convex operator. `lpd_flow` generates
a non-convex correction on top of a convex core. `map_gauss_newton` solves a Gauss-Newton
step weighted by the measured data covariance, with a shape prior covering the directions the
data does not constrain.
