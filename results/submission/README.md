# The submitted bodies

`Asteroid04.stl` to `Asteroid10.stl`, one per scored model, in the challenge pose: the
rotation axis is z, the body touches z = 1 and z = -1, the light is at minus infinity on x,
and the body stands as it did at frame 0 of the lightcurves. Each file is one watertight
component of positive volume with consistent winding whose largest distance from the axis
is the published bounding radius, which `scripts/check_submission.py results/submission`
verifies.

They are the convex stage's answers, written by `scripts/make_submission.py` from
`models/lpd_convex.pt` and the released Blender curves of each model, in seconds on a CPU.
The same recipe applied to the public models is under `results/public/`, and
`results/public_scores.json` holds their scores under the organisers' two measures together
with the checkpoint digest and the channel each model was inverted from. The README at the
repository root says why the convex answer is submitted and how a refinement would earn its
place; when `scripts/select_answers.py` has replaced any of these files, `selection.json`
beside them says which and by what margin.
