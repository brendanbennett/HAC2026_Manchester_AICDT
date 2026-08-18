# hac26
KNOWN ISSUES ORDER OF URGENCY:
- Flow Checkpoints: Currently corpus building (at current configuration takes 50s / body at 1000 bodies. This is not saved between corpus build and flow training or checkpointed throughout like surrogate building/training and is only saved when .stls are created . A failed flow training will not only lose the training but also the 14 hour corpus build.
- No early stopping: 4000 (configurable) steps in flow training will run irregardless of convergence. This is is computationally costly as each step draws from two bodies (from the 1000 created in the corpus). This can take on the order of days even with no loss improvement.
- Parallelisation: The remote pipeline has to run Curves_from_code which is called in a for loop = 50s (*1000 bodies in corpus building) + (2*50*4000 iterations in flow training). For Corpus building should be easily parallelisable as each body should be created independently, and some parallelisation is definitely possible in flow training as well.
- Data download checksum fails

UPDATE:
Previous corpus came from mesh booleans, which produced bodies that are two interpenetrating closed surfaces; a signed distance sampled against such a mesh is not a signed distance, so the codes fitted to it were fitted to noise. Build the bodies as level sets on an occupancy grid instead, where connectedness, void freeness and closure are all decidable and repairable before any triangle exists, and marching cubes returns a closed oriented manifold by construction.

- hac26/shape_library.py, curves_mesh.py, library_io.py, library_metrics.py: generation, mesh extraction, on-disk format and the acceptance metrics.
- scripts/build_shape_library.py: parallel, resumable generation to disk.
- scripts/fit_shapes.py: --shapes-dir to draw the corpus from a built library instead of train_surrogate.shapes().
- Thread --decoder-file through train_lpd, reconstruct_lpd and ablate_flow so a run is not pinned to runs/token_decoder.pt.
- scripts/_venv_setup.sh, run_remote_pipeline.sh, run_smoke_test.sh, validate_generation.py: remote run plumbing and generation checks.

Surrogate trained on: 
python scripts/train_surrogate.py --width 96 --blocks 3 --modes 8 --train 256 --held 16 --phases 32 --steps 6000

Main body trained with:
export N_BODIES=1000
export RECON_RES=64
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
