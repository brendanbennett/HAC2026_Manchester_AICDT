# hac26

Shape reconstruction from lightcurves for the
[Helsinki Asteroid Challenge 2026](https://fips.fi/data-challenges/helsinki-asteroid-challenge-2026/).

Ten 3D-printed asteroids were filmed on a turntable from 28 camera geometries. Each frame is
reduced to two numbers, summed intensity and lit-pixel count, giving 56 curves per body.
Reconstructions are scored on voxel overlap with the true shape plus the distance between the
boundary curves of 2D projections. `docs/challenge_info.md` has the rules.

## Reproducing the submission

The submitted reconstructions are `results/lpd/Asteroid01.stl` to `Asteroid10.stl`, in the
pose the challenge asks for. To rebuild them from the released data:

```
make venv toolchain        # the environment and nvdiffrast; see "Install" below
make data                  # the organisers' released data into dataset/raw
make submission            # the ten reconstructions into results/lpd
```

`make submission` does no training. It reads the two trained checkpoints committed under
`models/` -- `lpd_convex.pt`, the convex stage, and `lpd_flow.pt`, the flow that corrects it
-- and reconstructs each model in turn, a few minutes each on one GPU. It needs a CUDA GPU
and a working nvdiffrast; `make check` reports on both. Every mesh it writes is checked
against the challenge's pose and bounding-cylinder requirements before it exits, by
`hac26/submission.py`, which can also be run on its own:

```
PYTHONPATH=. python hac26/submission.py --dir results/lpd
```

Training those checkpoints is `make pipeline`, described under "Running the pipeline"; it
takes days on a GPU and is not needed to reproduce the results above.

`docs/results.md` has what the submitted reconstructions score on the three public
asteroids, against the convex stage alone as a baseline, and why the flow is the one
submitted. `METHOD=convex make submission` builds that baseline into `results/convex`.

## Install

`make` does the whole setup; `pyproject.toml` is the only dependency list.

```
make venv               # a .venv with the right Python and dependencies
make venv CUDA=12       # torch built against CUDA 12.x  (cu129 wheels)
make venv CUDA=13       # torch built against CUDA 13.x  (cu130 wheels)
make check              # what the venv has: python, torch, CUDA, nvdiffrast
```

Without `CUDA=`, the torch build is chosen from the driver `nvidia-smi` reports (CUDA 13
wheels need a 580-series driver or newer), and a machine with no GPU gets the CPU build --
on macOS that is the MPS build. Switching between 12 and 13 later is `make venv CUDA=13`;
it notices the installed torch is the wrong build and replaces it.

The Python is chosen by version, not by whichever `python3` comes first on PATH: the project
needs 3.10 or newer and cluster images often ship an older one as `python3`, which otherwise
produces a venv that fails much later. `uv`, if it is installed, is used and fetches a
matching interpreter itself; otherwise the newest suitable `python3.X` on PATH is used.
`make venv PY_VERSION=3.11` asks for a particular one, `make venv VENV=/scratch/env` puts it
elsewhere, and `make deps` reinstalls everything.

The exact forward model renders with nvdiffrast on a GPU; nvdiffrast is not on PyPI and
compiles CUDA at install time, so it is built from source against a toolkit assembled
without root:

```
make toolchain
```

That leaves it importable from the venv, with no `PYTHONPATH` to set. The calibration, the
flow training and the reconstruction all render with it. The tests run the same code on a
slow pure-torch rasteriser, so they need neither it nor a GPU.

It is compiled for the GPU architectures in `TORCH_CUDA_ARCH_LIST`, or, when that is unset,
for the card the build runs on. A build made on a newer card fails on an older one with "no
kernel image is available", so where one build serves several kinds of GPU, name them all:
`TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0+PTX" make toolchain`. The list is recorded with the build,
and changing it rebuilds.

`scripts/run_smoke_test.sh` and `scripts/run_remote_pipeline.sh` build the same venv through
the same Makefile if it is not there yet, so `make smoke` and `make pipeline`, or the scripts
by hand, are equivalent.

## Data

The challenge data is not stored in the repo. Fetch it with

```
python scripts/fetch_data.py
```

which streams the organisers' Dropbox folder into `dataset/raw/`, lifts out the wrapping
directory some releases have, and verifies what arrived. Override the link with `--url` or
`$HAC_DATA_URL`. Or put it there by hand under the organisers' original directory names.

Either way, check it with

```
python scripts/check_data.py
```

which verifies it against `dataset/MANIFEST.sha256` and names anything missing or changed.
Do that after every download: the organisers have re-released these files more than once, and
a partial refresh is silent. `--write` regenerates the manifest, for a refresh you meant to
make. `calibrate.py` reports a per-azimuth phase offset for the same reason -- the realignment
that came with the Aug 2026 update is a different shift per azimuth, which no single start
phase absorbs.

## Running the pipeline

Two wrappers do the whole thing:

```
scripts/run_smoke_test.sh        # wiring check: minutes, no dataset needed
scripts/run_remote_pipeline.sh   # the real run
```

`run_remote_pipeline.sh` builds and activates the venv through the Makefile if it is not
already there, then runs the stages in order:

| stage | script | notes |
| --- | --- | --- |
| `models` | `fetch_shape_models.py` | downloads public asteroid shape models; `FETCH_MODELS=0` skips it |
| `objects` | `fetch_objects.py` | everyday printable objects from Thingi10K; only with `FETCH_OBJECTS=1` (`pip install thingi10k`) |
| `library` | `build_shape_library.py` | CPU only; draws on the shape models and objects when there are any |
| `design` | `make_design.py` | GPU if there is one |
| `calibrate` | `calibrate.py` | needs `dataset/raw`; skipped if `models/instrument_calibration.pt` is present |
| `fit` | `fit_shapes.py` | fits shape codes to the library |
| `corpus` | `build_corpus.py` | renders every body and runs the convex stage on it; needs `models/lpd_convex.pt` |
| `prior` | `train_prior.py` | the prior part of the flow; no operator, minutes |
| `flow` | `train_lpd.py` | the data part, one expert on the straight line; needs the calibrated instrument |
| `flow-rollout` | `train_lpd.py` | the same run continued: branched into its experts, trained on the sampler's own states; the main phase |
| `decision` | `decision_check.py` | reconstructs the held-out corpus bodies and scores every rule for picking the answer against their truth |
| `convex` | `reconstruct.py` | the convex starts of all ten models; needs `dataset/raw` |
| `reconstruct` | `reconstruct_lpd.py` | all ten models, from those starts, each draw polished on the exact misfit |
| `score` | `hac26/scoring/` | the three public models |

Each stage writes a marker under `runs/.done/` recording the settings and the source it ran
with, and is skipped only while both still match, so a dropped run is safe to relaunch and a
change to the code reruns the stages below it on its own. Output goes to `logs/<stage>.log`.
Every setting is a variable at the top of the script and can be overridden from the
environment:

```
N_BODIES=2000 scripts/run_remote_pipeline.sh
scripts/run_remote_pipeline.sh --force-stage fit    # redo fit and everything after it
```

### On CSF3

`submit_csf3.sh` runs the whole pipeline as one Slurm job on the University of Manchester
CSF3. Run it from a login node; it submits itself:

```
./submit_csf3.sh                            # gpuA (A100 80GB), 4-day limit
CSF_PARTITION=gpuH_short ./submit_csf3.sh   # H200, 1-day limit
CSF_PARTITION=gpuH ./submit_csf3.sh         # H200, 4-day limit
CSF_PARTITION=gpuL ./submit_csf3.sh         # L40S 48GB, 4-day limit
./submit_csf3.sh -d afterany:1234           # other arguments go to sbatch
```

`#SBATCH` lines cannot read the environment, so the partition is passed on the `sbatch`
command line instead, together with what it needs: the H200 partitions take an account
(`gpu-h200-fse-pgdr`, or `CSF_ACCOUNT`), at most 8 cores per GPU rather than 12, and on
`gpuH_short` a wallclock of one day at most. `CSF_TIME` overrides the wallclock. Plain
`sbatch submit_csf3.sh` still works but always lands on gpuA; with `CSF_PARTITION` set it stops
at once instead of running on the wrong GPU. Pipeline variables such as `N_BODIES` pass
through the environment as usual.

The job loads its modules, puts caches, the challenge data, the shape models, the library
and `runs/` on `~/scratch/hac26` (`SCRATCH_DIR` to move it) because home is too small, builds
the venv and nvdiffrast for the A100, L40S and H200 alike, and then runs
`run_remote_pipeline.sh`. Its own output is in `logs/csf3_<jobid>.out`. A job that hits its
wallclock resumes where it stopped when resubmitted, so on `gpuH_short` the run is a chain of
one-day jobs:

```
J=$(CSF_PARTITION=gpuH_short ./submit_csf3.sh --parsable)
CSF_PARTITION=gpuH_short ./submit_csf3.sh -d afterany:$J
```

## Running stages by hand

Build the set of surface normals the shape representation is defined on. Once per size:

```
python scripts/make_design.py --n 4096
```

Fit the instrument to the real curves of the three public models, using their released
shapes. This writes `models/instrument_calibration.pt`, which training and reconstruction
require, and prints the residual of the exact forward model at the true shape divided by the
measured noise, per geometry -- the number that says how well the forward model matches the
organisers' processing, and which everything downstream rests on:

```
python scripts/calibrate.py
```

Build a shape library, fit codes to it, build the corpus, train the prior part of the flow
and then the data part:

```
python scripts/fetch_shape_models.py
pip install thingi10k && python scripts/fetch_objects.py --n 600
python scripts/build_shape_library.py --n 5000 --shape-models dataset/shape_models
python scripts/fit_shapes.py --shapes-dir dataset/generated/shapes --bodies 2000
python scripts/build_corpus.py
python scripts/train_prior.py
python scripts/train_lpd.py --steps 1000 --experts 1
python scripts/train_lpd.py --steps 1000 --extra-steps 1000 --rollout-frac 0.5
```

The library is what the secret bodies are likely to be: real asteroid shape models (the
radar and spacecraft models `fetch_shape_models.py` downloads, plus any OBJ, PLY or STL
dropped into the same directory), everyday printable objects from Thingi10K
(`fetch_objects.py`, into `dataset/shape_models/objects`), both stretched and mirrored at
random, and procedural bodies of the kinds real asteroids and test solids come in: smooth
lumpy potatoes, contact binaries with two or three lobes, spinning tops, faceted bodies,
boxes and prisms and cylinders with saw cuts; with basins, cuts, added lobes, ridges and
moderate roughness on top. Convex bodies are kept, so the flow also learns when there is
nothing to carve, and the mix of how deeply carved the bodies are is set directly
(`LibrarySpec.convexity_shares`: a quarter below 0.7 of their hull volume, a fifth at or
above 0.95), so the deeply carved tail is covered whatever the families would give on their
own. Each body is then mounted on one of its principal axes with a random
tilt, or at random, as the organisers mounted theirs, and the width over half-height that
gives is the radius it is rendered at (the number the challenge publishes per model); both
parts of the flow read that radius, and the operator poses every iterate the way the
challenge poses its models before rendering it. `dataset/generated/shapes/report.md`
summarises the library; `measure_public_shapes.py` compares it with the public bodies, and
`fit_shapes.py` reports per family how faithfully the shape code can hold its bodies, which
is the check that thin parts of an object are within the solver's reach.

The corpus is each body's curves from the exact forward model and the start the convex stage
(`models/lpd_convex.pt`, the same checkpoint `reconstruct.py` uses) reconstructs from a noisy
realisation of them; the flow is trained to make the correction from that start to the true
body, so it learns the errors the convex stage actually makes. The velocity of the flow is a
prior part, an unconditional flow over the corpus codes that reads no data and trains without
the operator, plus a data part that reads the residual and the adjoint of the exact forward
model and trains with the operator in the loop: every training step renders the body the
prior says the state is heading for and runs the adjoint back onto its code. The data part
is one reader, which interprets the curves and is shared by all times, and one expert per
interval of t, because the job changes along t: early the velocity has to come from the
curves, late it is a clean-up. Late in t the endpoint the velocity implies is rendered once
more and must fit the data to within the noise. The training data carry noise at the measured
level and a model-error term of the size the calibration fitted, and the residual is divided
by the two combined, as at reconstruction; every body is also turned by random quarter
turns about its spin axis, an exact symmetry the corpus carries the count curves for.
`build_corpus.py --phases` and `--operator-res` have to agree with `reconstruct_lpd.py`;
they share defaults, and the training scripts take them from the corpus.

`fit_shapes.py` and `train_lpd.py` both checkpoint and resume, so `--steps` is a cap and not
a schedule. `train_lpd.py` also early-stops on a held-out split (`--val-bodies`,
`--val-every`, `--patience`). Training is two runs of it: the first trains one expert on
states of the straight line between noise and body; the second continues its checkpoint,
copies the expert into one per interval of t (branching, `--experts`, four by default) and
trains half its draws on states the sampler itself reaches (`--rollout-frac 0.5`). The
second run is the main one; the pipeline runs both.

Reconstruct one model: the convex stage first, since its output is the start the flow
corrects, then the flow. Each draw is polished afterwards: gradient steps on the exact
misfit, stopped at the noise level, so the answer explains the data whatever the learned
steps left. `--hold-out-geoms 4` keeps four cameras away from the inversion and reports the
answer's misfit on them: the test of whether a shape was recovered rather than curves fitted.

```
python scripts/reconstruct.py --ckpt models/lpd_convex.pt --model 4 --out results/convex/Asteroid04.stl
python scripts/reconstruct_lpd.py --model 4 --out results/lpd/Asteroid04.stl
python scripts/reconstruct_lpd.py --model 1 --hold-out-geoms 4 --out results/check/Asteroid01.stl
```

Score against a public model:

```
PYTHONPATH=. python hac26/scoring/voxel.py --stl results/lpd/Asteroid01.stl --models 1
PYTHONPATH=. python hac26/scoring/side_view.py --models 1 2 3 --recon-dir results/lpd
```

The answer to a model is picked among the draws and their consensus bodies by expected
score against the draws, after rejecting candidates with non-finite misfit, open surfaces or
bad volume. Whether that rule beats the alternatives (the best-fitting draw, the medoid, a
fixed consensus level) can only be measured on bodies whose truth is known, which the
held-out corpus bodies are; the pipeline's `decision` stage does that and writes
`runs/decision_check.json`, including the best guidance weight and the gain over the convex
start in each carving bin:

```
python scripts/decision_check.py --bodies 16 --val-bodies 16
```

Check that the flow is using the lightcurves rather than memorising the corpus. This reruns
the flow loss on held-out bodies with real curves, shuffled curves and masked curves, and
prints the data-branch margin by time and carving bin:

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
hac26/forward/    forward models; see forward/__init__.py
hac26/solvers/    lpd_convex, lpd_flow, minkowski, output
hac26/scoring/    voxel, side_view
hac26/            conventions, geometry, field, shapes, noise, shape_library, curves_mesh,
                  library_io, library_metrics
scripts/          entry points
models/           trained convex solver and calibrated instrument; models/load.py loads them
results/          reconstructions: convex/ the convex stage's starts, lpd/ the flow's
dataset/          challenge data, not tracked
runs/             training output, not tracked
tests/
```

There are two forward models: an analytic operator for convex bodies, and the exact mesh
chain (shadows from a rasterised sun view, radiosity, rasterisation from every camera, the
sensor model), which is differentiable in the mesh. `hac26/forward/__init__.py` describes
both.

`lpd_convex` is an unrolled primal-dual network on the convex operator; its output is the
starting support of the main solver. `lpd_flow` is the main solver: a conditional flow that
generates a non-convex correction on top of that convex start, reading at every step the
residual of the exact model and its adjoint (`hac26/solvers/operator.py`).
