# What the depth field can hold

The body is a convex core displaced inward by a depth carried on directions
(`hac26/field.py`). How many directions it is carried on decides what a fit can reach, and
the question is not how well the family can approximate a shape but how well it can hold
the one shape the challenge actually poses: a body whose own hull is wrong in the places a
convex inversion cannot see.

## The target

The released non-convex body, model 3, posed into the canonical frame. Against its own
convex hull it has 0.768 of the volume and 0.869 of the overlap, so the whole of what any
correction has to supply is that 0.13 of overlap. The other two released bodies are their
own hulls to 0.997 and 0.9999 and say nothing about a correction at all; nothing here is
inferred from them.

## The fit that measures the family

The core is pinned to the body's own hull and the depths are solved, not descended to. What
is solved is an identity rather than an approximation: on the body's own surface the field
has to vanish, so the depth wanted in the direction of a surface point is exactly minus the
core's field there, which is the distance from that point out to the hull along the normal.
The fit is therefore a linear least-squares problem over points sampled by area over the
body's surface, with twelve points per coefficient and a ridge of 1e-3 of the mean diagonal,
and it needs no signed distance anywhere.

| directions | sample points | solve | residual | deepest | dice | components | condition |
|---|---|---|---|---|---|---|---|
| 640  | 7680  | 0.23 s | 0.0073 | 0.546 | 0.9923 | 1 | 91 |
| 2560 | 30720 | 0.66 s | 0.0037 | 0.557 | 0.9972 | 1 | 287 |

**The family is not what stands between a reconstruction and the body.** At 2560 directions
it holds model 3 to an overlap of 0.997, and the residual of the field over the body's own
surface is 0.004 in body units, a fortieth of the deepest carve it is holding. Every fit of
that body from its curves returns about 0.75. What is missing is not expressive power.

**640 directions already hold the body better than the lattice this replaced.** The lattice
of 13824 amplitudes reached an overlap of 0.991 on the same body; 640 depths reach 0.992 and
2560 reach 0.997, at a twentieth and a fifth of the coefficients. The two were fitted
differently -- the lattice against a signed distance in a band about the surface, because it
had to be -- so what the comparison settles is what each family can hold and not which fit is
better posed. The reason is not resolution but direction: a Gaussian in space smooths the carve
at the kernel's width along the surface normal as well as across it, and the normal is the
direction a shadow edge lives in, while a depth is smoothed across the surface only.

**The star-shaped bound is not slack and is not violated.** The deepest carve model 3 needs
is 0.557 against a bound of 0.570, so the body sits just inside it, and the extraction
returns one watertight component at every size above. That bound is what makes the single
closed surface `scripts/check_submission.py` requires a property of the representation rather
than something a repair pass has to achieve.

**The conditioning is a property of the sampling as much as of the node set.** At twelve
points per coefficient the normal matrix has a condition number of 91 at 640 directions and
287 at 2560; more points lower it. It is why the ridge is kept and why the point count is
set per coefficient rather than per body.

## The node set, and why it is not a plain spiral

A quarter turn about the spin axis is an exact symmetry of the problem and is what gives the
corpus four training pairs per body at no cost in renders. That is only an exact symmetry of
the *code* if turning a body permutes its depths. A plain Fibonacci spiral is not invariant
under a quarter turn, so a turn would have to resample the field instead -- and resampling
smooths it. Measured on model 3, the resampled code differs from a fit of the turned body by
twelve per cent of the depths in the root mean square, which is five times the fit's own
residual, and by forty per cent of the deepest carve at the worst direction. A corpus
augmented that way would teach the flow codes that no fit of those bodies would produce.

The node set is therefore a quarter of the sphere's worth of directions and its three
rotations, and the quarter carries a spiral of its own: latitudes evenly spaced in cosine,
azimuths advancing by the golden ratio *of the quadrant*. Advancing instead by the whole
sphere's golden angle and folding it into the quadrant leaves consecutive latitudes near the
poles at nearly the same azimuth, and the closest pair of nodes then sits at half the spacing
of a plain spiral. Taking the golden ratio inside the quadrant costs nothing:

| node set | nearest neighbour, in units of the mean spacing | | | sixth neighbour |
|---|---|---|---|---|
| | min | median | max | max |
| plain spiral | 0.872 | 0.950 | 0.990 | 1.590 |
| quarter-turn invariant | 0.852 | 0.954 | 0.988 | 1.518 |

The turn is then a shift of whole blocks, exact to machine precision.

## The extraction

The surface is extracted by FlexiCubes on a grid of `EXTRACT_RES` cubes over
`[-EXTRACT_EXTENT, EXTRACT_EXTENT]`. Two things about it are worth recording because they
decide whether a fit of thousands of renders is affordable at all.

The grid itself is built arithmetically rather than by deduplicating the corners of every
cube separately. The vendored construction sorts eight times as many points as the grid has,
which at 128 cubes a side is seventeen million rows of three and several gigabytes; it does
not complete in a seven-gigabyte machine. The arithmetic construction agrees with it exactly,
vertices and cube corners alike, and the suite checks that at four sizes.

The core's field and the depth's neighbour search are both cached on the identity of the
points they were computed for. Through a fit of the carve the support does not move and the
grid is one tensor, so both are computed once and reused on every later render. Measured on
one core pair: a first extraction at 88 cubes a side costs 14 s and every later one 0.36 s,
and at 128 cubes 38 s and 0.95 s. Without the caches every render pays the first figure,
which is the difference between a fit that takes half an hour and one that takes a day.
