# Whether the curves choose the body

The non-convex path minimises the misfit between rendered and released curves. That is only
worth doing if the body has a lower misfit than the wrong bodies near it, and if the search
can get from where it starts to where the body is. Those are two different questions and they
have different answers. This note measures both on model 3, the one released body with a
genuine concavity, and states what the answers require of the forward model.

## The measurement

The convex stage's answer for model 3 supplies the base support. The body is written in the
model's own coordinates: the nine reshaping coefficients that carry that answer's hull onto
the body's own convex hull, solved on points sampled from that hull, and the lattice
amplitudes that carve the rest, solved by weighted ridge least squares against what is left of
the body's signed distance. Curves are rendered through an orthographic ray cast with a
Lambert surface, cast shadows and a power-law transfer, at 24 phases and 96 pixels over the 21
distinct geometries and both curve types. `chi` is their root mean square difference from the
released Blender curves; `dice` is the voxel overlap with the released shape at 128 cubed.

This stand-in renderer reproduces the released curves at the released mesh to a root mean
square of 0.011. That number is the scale everything below is measured against, and on the
exact chain it is what `scripts/calibrate.py` reports.

| body | chi | dice |
|---|---|---|
| the convex stage's answer | 0.0782 | 0.7075 |
| the fit, from that answer, on every camera | 0.0376 | 0.7205 |
| the body itself, written in these coordinates | 0.0349 | 0.9888 |
| the fit, released from the body | 0.0307 | 0.9584 |

## What it says

**The objective is the right one.** Released from the body the fit does not run away: it stays
within 0.03 of the overlap it started with and lowers the misfit, and the misfit it reaches,
0.0307, is below the 0.0376 the fit reaches from the convex answer. The body's neighbourhood
is a basin of the misfit and it is the deeper of the two. Nothing here asks for a different
functional to minimise, or for a prior to break a tie between the body and a wrong one; the
curves do prefer the body.

**The search does not cross into it.** From the convex answer the fit takes the misfit
slightly more than halfway to the body's own and the overlap almost nowhere. It stops in a
basin that is not the body's, and it stops there while still descending. What separates the
two runs is not the objective, the representation or the step: it is which basin the start was
in.

**The two basins are separated by less than the forward model's error.** The misfit at the
bottom of the wrong basin exceeds the misfit at the bottom of the body's by 0.0069, while the
renderer used here misses the released curves at the released mesh by 0.011. A comparison of
misfits at that accuracy cannot be trusted to prefer the body over the wrong body by the
margin that separates them, however good the optimiser. The ordering measured above is real,
but it is finer than the instrument that measured it.

That last point is the one to act on, and it is not a statement about this stand-in. The same
subtraction applies to the exact chain: whatever residual `scripts/calibrate.py` reports at
the released shapes has to be well below the gap between basins, or a search that selects on
the misfit is selecting inside its own error. Read that residual before reading any
reconstruction, and read the misfit on the held-out cameras rather than the fitted ones, which
is what `scripts/select_answers.py` requires and why it will refuse a correction that no
public body's correction has vouched for.

## What the lattice contributes

The same fit run on a lattice of twelve sites per axis instead of twenty-four reached a misfit
of 0.0420 at an overlap of 0.7095 — that is, it fitted the curves *better* than the best that
lattice can do at the body itself, which is 0.0513 (`representation.md`). A representation too
coarse to hold the body does not merely limit the answer; it lets a wrong body outrank the
body on the data, and then the ordering above is not merely finer than the instrument, it is
reversed. At twenty-four sites the fit lands above the body's own value rather than below it,
which is the condition under which the misfit means anything at all.
