# Results

What the two reconstruction methods in this repository score, and why the flow is the one
submitted.

The challenge scores a reconstruction two ways: a voxel overlap measure, higher is better
and 1 is perfect, and a side-view distance between the boundary curves of 2D projections,
lower is better. Only asteroids 1-3 are public, so those are the only real bodies either
method can be scored against here; asteroids 4-10 are the scored, secret ones.

## The public asteroids

Reproduced by `make submission` and scored by `hac26/scoring/voxel.py` and
`hac26/scoring/side_view.py`. The flow is sampled at guidance 2.0, the weight the decision
check chose.

| model | convex voxel | flow voxel | convex side ASSD | flow side ASSD |
| --- | --- | --- | --- | --- |
| 1 | 0.9800 | **0.9816** | 0.0122 | **0.0109** |
| 2 | **0.8975** | 0.8926 | **0.0671** | 0.0680 |
| 3 | 0.6914 | **0.7089** | 0.1139 | **0.1038** |
| sum / mean | 2.5689 | **2.5831** | 0.0644 | **0.0609** |

The flow is ahead on both measures in aggregate: +0.0142 summed voxel and -0.0035 mean
side-view distance. It is behind on model 2 alone.

That is not a coincidence of which model. Two reference rows from the side-view scorer say
how non-convex each body actually is:

| model | truth vs itself | convex hull of the truth |
| --- | --- | --- |
| 1 | 0.0003 | 0.0011 |
| 2 | 0.0004 | 0.0004 |
| 3 | 0.0002 | 0.0386 |

The first is the floor set by sampling and pixel resolution. The second is what a perfect
convex reconstruction would score, so it measures the carve the convex stage cannot reach:
asteroid 2 is the most nearly convex of the three, asteroid 1 next, asteroid 3 by far the
least. Order the flow's voxel gain the same way and it is monotone:

| body | hull vs truth | flow voxel gain |
| --- | --- | --- |
| 2, most convex | 0.0004 | -0.0049 |
| 1 | 0.0011 | +0.0016 |
| 3, least convex | 0.0386 | +0.0175 |

The flow adds carving. It costs a little on a body that has none to add and gains on a body
that does, which is the behaviour the method is designed to have, and it is why the one loss
falls on the one body with nothing to carve.

## Held-out corpus bodies

The guidance weight was chosen against synthetic bodies from the training corpus,
reconstructed exactly as the challenge models are (`scripts/decision_check.py`, four held-out
bodies, `runs/decision_check.json`):

| guidance | flow Dice | convex Dice | gain |
| --- | --- | --- | --- |
| 1.0 | 0.7192 | 0.7262 | -0.0070 |
| **2.0** | **0.7357** | 0.7262 | **+0.0095** |
| 3.0 | 0.7304 | 0.7262 | +0.0042 |

Split by how carved the body is, the same pattern holds at every guidance weight, which is
what makes the public-asteroid ordering above a prediction rather than a rationalisation:

| guidance | low carve | medium carve | high carve |
| --- | --- | --- | --- |
| 1.0 | -0.0199 | -0.0445 | +0.0181 |
| 2.0 | -0.0019 | -0.0146 | +0.0273 |
| 3.0 | -0.0081 | -0.0149 | +0.0198 |

## How much to trust the margin

The flow's lead on the public asteroids is +0.55% of the summed voxel measure over three
bodies. That is thin, and worth stating plainly:

- It is consistent across two independent measures and across two independent body sets, the
  three real asteroids and the four held-out corpus bodies.
- It has a mechanism, and the mechanism predicts the sign of the per-model difference from a
  quantity measured before the comparison.
- It is nonetheless three bodies, and the scored set is seven secret ones. If those are all
  as nearly convex as asteroid 2, the convex stage would have been the better submission.
  `METHOD=convex make submission` rebuilds it.

An earlier version of this file recommended the convex stage. That was based on scoring a
flow trained without the rollout phase and sampled at guidance 1.0, which summed 1.9948 --
the stage that scores the meshes had a cache key that did not cover either setting, so it
skipped, and the stale numbers read as though they described the current model. The cache
key is fixed in `scripts/run_remote_pipeline.sh`; the numbers above are the current model.

## Reproducing

```sh
make venv toolchain
make data
make submission        # the ten submitted STLs into results/lpd
```

Scoring the public models needs the released data and the truth STLs it contains. Neither
scorer uses a GPU:

```sh
PYTHONPATH=. python hac26/scoring/voxel.py --stl results/lpd/Asteroid0{1,2,3}.stl --models 1 2 3
PYTHONPATH=. python hac26/scoring/side_view.py --models 1 2 3 --recon-dir results/lpd
```
