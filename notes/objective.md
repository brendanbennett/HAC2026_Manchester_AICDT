# What is minimised, and why it is not the misfit

A reconstruction is scored on how much of the body it overlaps. It is fitted on how well its
curves match. Those are different functionals and this note measures how differently they
rank the same bodies, which is what decides whether minimising the second can ever return a
body that scores on the first.

## The misfit reports sharpness as much as shape

Take the convex stage's answer for model 3 and move it in two ways. One is toward the body:
blend the answer's signed distance with the body's, which changes the gross shape and leaves
the surface as smooth as it was. The other is a corrugation: add a random field of one grid
cell's width and a few hundredths of a body radius in amplitude, which leaves the gross shape
where it is.

Measured, through an orthographic ray cast with the measured photometry, at 24 phases and 96
pixels over the 21 distinct geometries: every corrugation of two to four hundredths lowers the
misfit by 0.013 to 0.023 and moves the overlap by 0.003. Moving thirty per cent of the way to
the body lowers the misfit by 0.011 and moves the overlap by 0.081. So a corrugation buys more
misfit than the right direction does and buys almost no overlap with it.

The reason is that both reductions are sums over a thresholded image. Their error is the
quantisation of the boundary of the lit region, and a finer surface has more boundary to place
more accurately. Nothing about that is a property of this renderer; the organisers' own
reduction has it.

## Area is the functional that separates them

The two directions differ in one measurable way. A corrugation raises the body's surface area
by 0.55 to 0.71; moving toward the body lowers it by 0.38 to 0.72. The surface area of a
closed body is the total variation of its indicator, and it is what charges a rough surface
while leaving a smooth dent of any depth nearly free. A ridge on the amplitudes does not: it
charges by how large a coefficient is, so it prefers a shallow answer to a deep one, and
measured against the released body it prefers the fit's own answer by eight per cent.

## The weight has to be scale free

Under misfit-squared plus a weight times area, the released body is the minimiser of a span of
fifteen posed bodies only for weights at or below 6.6, while a corrugation one cell wide is not
charged until the weight is above 34.5. There is no fixed weight that does both; the window is
empty by a factor of five.

The reason is that the gradient of the squared misfit falls with the misfit while the gradient
of an area term does not. Making the objective scale free in the misfit,

    G = log chi^2 + mu A,

fixes the balance: stationarity then compares a relative change of misfit with an absolute
change of area. The window opens to 0.60 <= mu <= 0.99 — below 0.60 the one-cell corrugation is
still profitable, above 0.99 the body is beaten by itself displaced 0.03 inward, which costs
0.016 of overlap against the 0.07 the corrugation costs. `AREA_WEIGHT` sits near the top of
the window and `CarveFit` refuses a value outside it.

The penalty does not forbid concavities, and that was tested rather than argued. A hemispherical
pit of radius 0.15 cut into the released body at 35 places over its surface costs between
0.004 and 0.073 of objective and is paid between 0.005 and 0.964 by the misfit: 33 of the 35
are paid for, by a median factor of ten. The two that are not are places where the pit changes
the curves by less than the model error, which the data do not determine in any case.

## What the penalty exposed

Three things had to be added with it, and each is a defect the ridge was hiding rather than a
price of the penalty.

A direction the curves cannot see has no curvature in the Gauss-Newton matrix, so damping
relative to that curvature does not bound a step into it. While the right-hand side was the
Jacobian applied to the residual, it lay in the range of the Jacobian and the question did not
arise. The area's gradient does not lie there, and the amplitudes then grow without limit in
directions that do not move the surface: measured, a first coarse step reached a carve
seventeen hundred body radii deep, while rendering a body of overlap 0.748. Flooring the
damped diagonal at a hundredth of its mean bounds it, and the same step then reaches a carve
of 0.055 and the same body.

The cheapest area in this representation is a hull shrink. A step has to be shortened to stay
inside a trust region rather than merely refused by it, or a coarse stage collapses the body
and every trial is refused; the volume's and the carve depth's own secant derivatives say how
long a step may be, and both are needed, since a step that carves in one place and fills in
another leaves the volume where it was.

And a body whose convex answer already explains its curves has no concavity to find. There the
penalty trades overlap for a misfit the body does not need, and it does so while improving the
misfit, so nothing downstream can catch it. The ratio of the convex answer's misfit to the
channel's model error is read before the fit instead: it is 7.9 on model 3 and 1.8 on model 1,
and running the penalty on model 1 costs 0.166 of overlap.

## What it is worth

The numbers on model 3, from the convex stage's answer, through the same stand-in renderer,
are in the table below; the branch's own previous answer is the row marked as the ridge.

| what was minimised | misfit | overlap | carve depth | hull shift |
|---|---|---|---|---|
| nothing: the convex stage's answer | 0.0787 | 0.7080 | 0 | 0 |
| the misfit, under a ridge on the amplitudes | 0.0376 | 0.7205 | 0.224 | +0.016 |
| the misfit and the area, then polished on the misfit | 0.0557 | 0.7533 | 0.226 | +0.042 |

Both fits are from the same start, on the same curves, through the same renderer, at 24 phases
and 96 pixels. The penalised one reaches an overlap 0.045 above the convex answer where the
other reaches 0.013, and its hull correction is three times larger and in the direction the
body needs, the body's own being +0.087.

It reaches that at a *worse* misfit, which is the whole point and the thing to watch. A
selection that reads the misfit alone prefers the second row to the third and would throw the
better body away, so `scripts/select_answers.py` reads the functional the correction was
fitted under. Nothing that compares two corrections may read the misfit on its own.

The volume falls from the convex answer's 2.52 to 1.27 against the body's 1.46, so the penalty
overshoots the shrink by about a seventh of the volume even with the trust region holding each
step. That is the direction its risk was known to lie in and it is the first thing to look at
if the overlap stops improving.
