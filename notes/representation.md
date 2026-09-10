# What the correction lattice can hold

The body is a convex core plus a correction on a lattice of Gaussian kernels
(`hac26/field.py`). How fine that lattice is decides what a fit can reach, and the
question is not how well a lattice can approximate a shape but how well it can
approximate the shape's *curves*. Those are not the same, and the difference is what
this note measures.

## The measurement

Model 3 is the one released body with a genuine concavity. Its carve was laid on the
convex stage's answer for it as a field: the difference between the body's signed
distance and the polytope core of that answer, smoothed by three quarters of a grid
cell so that a level set of it is a surface rather than a staircase. That field is the
target. Each lattice was fitted to it by weighted ridge least squares on 150000 points
within 0.30 of the surface, weighted by `exp(-(d/0.12)^2) + 0.02` in the signed
distance, the weighting `scripts/fit_shapes.py` uses. The fitted field was added back
to the core, the zero level extracted on a 96-cubed grid over a half-width of 1.30 and
decimated to 12000 faces, and the body rendered at 24 phases and 96 pixels through an
orthographic ray cast with a Lambert surface, cast shadows and a power-law transfer.

`chi` is the root mean square difference between those curves and the released Blender
curves of model 3, over the 21 distinct geometries and both curve types. `dice` is the
voxel overlap with the released shape at 128 cubed. For reference, on the same
renderer and the same grids the released truth itself scores chi 0.011 and the convex
stage's answer scores chi 0.078 at dice 0.708.

The solve is iterative on a kernel truncated at four standard deviations, where it is
3e-4 of its peak. Against a dense solve of the same normal equations at 12 cubed it
agrees to 0.0004 in chi and 0.0000 in dice.

## The table

| sites per axis | sigma / spacing | ridge | chi | dice | largest amplitude |
|---|---|---|---|---|---|
| 12 | 0.90 | 1e-2 | 0.0513 | 0.9829 | 0.119 |
| 12 | 0.90 | 1e-4 | 0.0449 | 0.9865 | 0.355 |
| 16 | 0.90 | 1e-2 | 0.0399 | 0.9904 | 0.103 |
| 16 | 0.90 | 1e-4 | 0.0392 | 0.9076 | 0.281 |
| 24 | 0.75 | 1e-2 | 0.0258 | 0.9907 | 0.098 |
| 24 | 0.75 | 1e-4 | 0.0260 | 0.9912 | 0.156 |
| 24 | 0.90 | 1e-2 | 0.0290 | 0.9912 | 0.068 |
| 24 | 0.90 | 1e-4 | 0.0265 | 0.9914 | 0.182 |
| 32 | 0.75 | 1e-2 | 0.0265 | 0.9921 | 0.098 |

## What it says

**The curves see the sharpness, the overlap does not.** Between 12 and 24 sites per
axis the overlap with the truth moves by 0.008 and the misfit halves. A body at dice
0.983 that the coarse lattice can hold is twice as far from the curves as a body at
dice 0.991 that the fine one holds. Anything that selects on the misfit is therefore
choosing between representations far more sharply than the score would.

**That is the margin an inversion works in.** The convex answer sits at 0.078. On the
coarse lattice the best carve the representation can hold sits at 0.051, a factor of
1.5 below it; on the fine lattice it sits at 0.026, a factor of 3.0. The fine lattice
does not make the answer better by much, it makes the answer *distinguishable* from
the convex body it started as.

**Past 24 there is nothing left.** Thirty-two sites per axis reach the same misfit as
twenty-four and cost nearly three times the amplitudes. The kernels are then narrower
than the extraction grid resolves, and the limit stops being the representation.

**The ridge is not a free parameter.** At 16 sites per axis, loosening it from 1e-2 to
1e-4 lowers the misfit by 0.0007 and costs 0.083 of overlap: the fit buys a slightly
better curve match with a rough surface, which is the same trade a solver would make
if it were allowed to. At 24 sites the same loosening costs nothing, so the finer
lattice is better conditioned in the shape as well as finer. The default stays at the
tighter value, because what a scored body needs is the shape.

**A kernel narrower than the extraction pitch is not there.** At `EXTRACT_EXTENT` and
96 grid points the pitch is 0.033 against a kernel width of 0.069, so the grid carries
about two samples per kernel. `field.kernel_pitch_ratio` reports it, and an extraction
below one samples a body coarser than its own amplitudes describe.
