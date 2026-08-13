# hac26

Shape reconstruction from lightcurves for the
[Helsinki Asteroid Challenge 2026](https://fips.fi/data-challenges/helsinki-asteroid-challenge-2026/).

Ten 3D-printed asteroids were filmed on a turntable from 28 camera geometries. Each frame is
reduced to two values, summed intensity and lit-pixel count, giving 56 curves per body.
Reconstructions are scored on two measures, summed: voxel overlap with the true shape, and the
distance between the boundary curves of 2D projections. See `docs/challenge_info.md` for the
full rules.

## Installation

```
pip install -e ".[torch]"
```

Only the mesh rasteriser needs a GPU (nvdiffrast). Everything else runs on CPU.

## Data

Download the challenge data to `dataset/raw/`, keeping the original directory names, and check
it against `dataset/MANIFEST.sha256`. About 5 GB, not stored here.

## Usage

Score a reconstruction against a public model:

```
PYTHONPATH=. python hac26/scoring/voxel.py --stl results/lpd/Asteroid01.stl --models 1
PYTHONPATH=. python hac26/scoring/side_view.py --models 1 2 3 --recon-dir results/lpd
```

Reconstruct one model, with the convex solver or with the flow:

```
python scripts/reconstruct.py --ckpt models/lpd_convex.pt --model 4 --out results/convex/Asteroid04.stl
python scripts/reconstruct_lpd.py --model 4 --out results/lpd/Asteroid04.stl
```

Retrain the flow solver. The first step builds the set of surface normals the shape
representation is defined on and only needs running once per size:

```
python scripts/make_design.py --n 4096 --device cuda
python scripts/fit_shapes.py --bodies 2000
python scripts/train_lpd.py --steps 4000
```

## Structure

```
hac26/forward/    forward models, one per file; see forward/__init__.py
hac26/solvers/    lpd_convex, lpd_flow, minkowski, map_gauss_newton, curve_loss, output
hac26/scoring/    voxel, side_view
hac26/            conventions, geometry, field, shapes, calibrate, noise, covariance
scripts/          entry points
models/           trained convex solver, instrument calibration; load with models/load.py
results/          reconstructions: convex/ is the submitted set, lpd/ the flow solver's
dataset/          challenge data, not tracked
runs/             training output, not tracked
tests/
```

Forward models are interchangeable and differ in shape parameterisation and physics, from an
analytic convex operator with no shadowing to a full mesh chain with interreflection and a
camera model. `hac26/forward/__init__.py` lists what each one covers and leaves out.

`lpd_convex` is an unrolled primal-dual network on the convex operator. `lpd_flow` generates a
non-convex correction on top of a convex core. `map_gauss_newton` solves a Gauss-Newton step
weighted by the measured data covariance, with a shape prior covering the directions the data
does not constrain.
