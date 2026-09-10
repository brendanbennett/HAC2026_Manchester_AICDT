# hac26

Shape reconstruction from lightcurves for the
[Helsinki Asteroid Challenge 2026](https://fips.fi/data-challenges/helsinki-asteroid-challenge-2026/).

Ten 3D-printed asteroids were filmed on a turntable from 28 camera geometries. Each frame is
reduced to two numbers, summed intensity and lit-pixel count, giving 56 curves per body, and
the organisers release both the laboratory curves and a Blender rendering of the true shape
under the same geometries. Reconstructions are scored on voxel overlap with the true shape
plus the distance between the boundary curves of 2D projections. `docs/challenge_info.md` has
the rules.

## The submission

The submitted bodies are the convex stage's answers, `results/submission/Asteroid04.stl` to
`Asteroid10.stl`, produced by

```
python scripts/make_submission.py
```

which reconstructs every model with the trained convex network `models/lpd_convex.pt`, poses
it as the challenge asks (rotation axis z, the body touching z = 1 and z = -1, the light at
minus infinity on x, frame 0 of the curves), sets its width from the published bounding
radius, writes the public models to `results/public/`, checks every file
(`scripts/check_submission.py`, which requires one watertight component of positive volume,
consistent winding, the pose and the cylinder) and scores the public ones against the released shapes
with the organisers' two measures (`hac26/scoring/official.py`) into
`results/public_scores.json`. It takes seconds per model on a CPU and needs neither a GPU
nor nvdiffrast. The recipe is fixed in the script, and `results/public_scores.json` records
the checkpoint's digest and the channel each model was inverted from.

The convex network reads the Blender curves when the organisers release them and the
laboratory curves otherwise (`hac26.data_io.load_inversion_curves`). The render is the
cleaner observation of the shape. It has no lens, sensor, mounting or beam in front of it
and no per-column realignment behind it, its scattering is of the kind the convex operator
assumes, and the laboratory columns of several bodies are out of phase with their own
geometry by tens of degrees. `scripts/reconstruct.py --channel` forces either channel for one
model.

Why the convex answer and not a carved one is measured rather than assumed. On the three
public bodies, one of which is a contact binary, every refinement of the convex answer that
this repository or its history has produced, the trained flow, gradient descent on the exact
misfit, searches over carved bodies, scored below the convex answer or equal to it under the
organisers' measures, because a body that fits the curves better than the convex answer is
not, on the evidence of the public bodies, a body closer to the truth. `scripts/benchmark.py`
scores any directory of reconstructions the same way, so a refinement earns its place by
that number.

```
python scripts/benchmark.py results/public results/lpd
```

## Install

```
pip install -e ".[torch]"
```

The exact forward model renders with nvdiffrast on a GPU. nvdiffrast is not on PyPI and
compiles CUDA at install time, and `scripts/setup_toolchain.sh` builds it without root. The
calibration, the flow training and the non-convex reconstructions render with it. The
submission and the tests do not, since the tests run the same code on a slow pure-torch
rasteriser.

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
Do that after every download, because the organisers have re-released these files more than
once and a partial refresh is silent. `--write` regenerates the manifest, for a refresh you meant to
make.

## The non-convex path

Everything beyond the convex stage is research code. It runs, it is tested, and none of its
answers are submitted, for the reason above. It needs an instrument fitted to the channel it
inverts, which `scripts/calibrate.py` writes from the public models' released shapes.

```
python scripts/calibrate.py                                # the laboratory channel
python scripts/calibrate.py --channel blender --models 1 3 # the Blender render
```

The two channels are different instruments (`hac26/forward/mesh/instrument.py`). The
laboratory curves come through a lens, a sensor and bounce light off a matte white print,
while the render has none of those and a far camera, so it starts from `Instrument.blender_start`,
a far camera with the interreflection switched off and an sRGB-like transfer curve, and is
written to its own file. The calibration prints the residual of the exact forward model at
the true shape divided by the noise, per geometry, which is the number that says how well
the chain matches the channel, and a travel table that names any parameter still moving
when the step budget ran out.

Descent from the convex answer, with cameras held out as the check on whether a shape was
recovered rather than curves fitted.

```
python scripts/reconstruct_map.py --model 3 --channel blender --hold-out-geoms 6 \
    --out results/map/Asteroid03.stl
python scripts/select_answers.py --refined results/map --ratio <measured on model 3>
```

`reconstruct_map.py` records, for the body as written, its misfit on the fitted and on the
held-out geometries beside the convex answer's, and on a public model the Dice against the
released shape at every checkpoint. `select_answers.py` replaces a convex answer only where
the refined body beats it on the held-out geometries by a ratio measured on the public
model whose refinement raised the score, and writes `results/submission/selection.json`
saying what it did.

The flow pipeline, `scripts/run_remote_pipeline.sh`, builds a shape library, fits shape codes
to it, renders a corpus with the exact operator and the convex stage's starts, trains the
prior and the data part of a conditional flow (`hac26/solvers/lpd_flow.py`,
`scripts/train_lpd.py`) and reconstructs every model from several draws with a polish on the
exact misfit (`scripts/reconstruct_lpd.py`). Each stage writes a marker under `runs/.done/`
recording its settings and the source it ran with, and is skipped only while both still
match, so a dropped run is safe to relaunch and a change to the code reruns the stages below
it. `scripts/run_smoke_test.sh` runs the same stages small, as a wiring check. Every setting
is a variable at the top of the pipeline script and can be overridden from the environment.
Training states lie on the straight line between noise and body only, and `train_lpd.py`
says why states from the sampler's own trajectory are not scored. The flow's answers are scored
beside the convex ones by the pipeline's last stage and are not part of the submission.

## Tests

```
pytest
pytest -m "not slow"     # skip the extraction-scale ones
```

## Layout

```
hac26/forward/    forward models; see forward/__init__.py
hac26/solvers/    lpd_convex, lpd_flow, minkowski, operator, output
hac26/scoring/    official (the organisers' measures), voxel, side_view
hac26/            conventions, geometry, field, shapes, noise, shape_library, curves_mesh,
                  data_io, recon, library_io, library_metrics
scripts/          entry points; make_submission.py is the submission
models/           the trained convex solver; the calibrations calibrate.py writes go here
results/          submission/ the scored models, public/ the public ones, public_scores.json
dataset/          challenge data, not tracked
runs/             training output, not tracked
tests/
```

There are two forward models, an analytic operator for convex bodies and the exact mesh
chain (shadows from a rasterised sun view, radiosity, rasterisation from every camera, the
sensor model), which is differentiable in the mesh. `hac26/forward/__init__.py` describes
both.

`lpd_convex` is an unrolled primal-dual network on the convex operator, trained on synthetic
convex bodies, and its output is the submitted body and the starting support of the non-convex
path. `lpd_flow` is a conditional flow that generates a non-convex correction on top of that
convex start, reading at every step the residual of the exact model and its adjoint
(`hac26/solvers/operator.py`).
