# Forward-model review, cross-referenced against Helsinki-Challenge-2026

Review of `hac26/forward/` and the calibration that fits it, done by cross-referencing an
independently developed forward model for the same challenge
(`Helsinki-Challenge-2026`, a Lambertian mesh renderer validated against the published
Blender curves). The two codebases share nothing but the challenge description, so
agreement between them is evidence and disagreement is a place to look.

Everything below is either a numerical comparison between the two models or a measurement
on the released data (`HAC_data_May_8`). Reproduction notes are at the end.

---

## What checks out

**The geometry conventions are exactly right.** `conventions.camera_vector`, `to_body` and
`psi_grid` (SENSE = −1, PSI0 = 0, δ = +1) were compared against the reference model's
independently derived camera and light directions over all 28 geometries × 360 phases:

```
lab camera directions    max |hac26 − reference| = 5.6e-16
body camera directions   max |hac26 − reference| = 1.6e-15
body sun directions      max |hac26 − reference| = 8.9e-16
conventions.to_body vs exact.rotate_z             2.2e-16
```

Light at `(-1, 0, 0)`, camera at lab azimuth `180 + az`, elevations per the published
table, frame *k* rotated by `R_z(+k·360/F)`, phase zero at the STL pose. The reference
model's best-fit phase offset against the Blender curves is 0, and `calibrate.py` fits
ψ₀ = 0.3°, 1.3°, 2.5° on the three public bodies — two independent confirmations that
`PSI0 = 0` is right.

**The transport chain is correct, and better than the reference model's in three places.**

- `LitCoverage`: `e_i = coverage / A_i = (n·s)₊` read off an orthographic sun raster is
  exact, and gets cast shadows, self-shadowing and penumbra for free. The clip-space
  *z* sign in `orthographic()` gives the correct depth test (nearer the source ⇒ smaller
  *z*).
- Radiosity: `B = (I − ρF)⁻¹ ρE`, `L = B/π`, direct light at mesh resolution and
  interreflection at patch resolution, is the right decomposition; the reference model has
  no interreflection at all.
- Radiance as a view-independent per-face attribute, with no 1/r² factor and foreshortening
  realised by pixel coverage, is right for a Lambertian surface.
- The coarea derivatives of the two thresholded reductions are exact where the reference
  model resorts to a soft threshold.

**Otsu computed from the render is the more faithful choice, and the obvious objection
against it does not hold.** The reference model fits a fixed binary threshold instead; the
worry with computing Otsu on a rendered frame is that the render has a black background and
a frame fill fixed by `fov_scale`, neither of which matches the video. Measured on
synthetic limb-darkened frames through `raster.otsu_threshold`:

```
frame fill 0.005 → 0.25       counted fraction of the body varies by  < 5%
background level/noise up to 0.10 ± 0.03   counted pixels vary by     < 0.5%
```

So the count is robust to both, and the argument in `Instrument.pedestal` for not fitting a
binary pedestal is sound. No change needed.

---

## 1. `sigma_from_replicates` is not measuring measurement noise

`hac26/noise.py` states: *"At each azimuth two cameras sit at the same place and see the
same body at the same instant, so their difference is measurement noise with no geometry in
it."*

That is not what the two horizontal columns are. Each public model ships **28 real videos**
— `CAM1/CAM2 × orientation 1A/1B × 7 angles` — and only **21 simulated** ones
(`{0,45,90,135,225,270,315} × {top, center, bottom}`). There were only ever two cameras, one
horizontal and one looking down. So `hor_a` and `hor_b` are the *same* nominal geometry
recorded in two different mountings of the body, aligned afterwards by time reversal and a
temporal shift (the challenge text describes exactly this). They are not simultaneous, and
their difference carries the A/B mounting mismatch, the residual alignment error and the
stem, not just noise.

Measured on the released curves — σ from the pair difference, against a per-curve
high-frequency estimate from successive differences, plus how much of the difference
survives a 15-frame box smooth:

```
model 3 intensity   σ(a−b)/√2   HF noise   ratio   low-freq share of (a−b)
  az  45              0.0048      0.0017     2.9          0.98
  az  90              0.0088      0.0028     3.2          0.97
  az 135              0.0318      0.0070     4.6          0.98
  az 225              0.0328      0.0061     5.4          0.98

model 1 intensity
  az 135              0.0517      0.0040    13.1          0.99
  az 225              0.0539      0.0027    20.0          0.99
```

σ is inflated by 3–20×, and 86–99% of the difference is low-frequency — structure, not
noise. Four consequences, all downstream of the same number:

- The `per_sigma` column of the residual report is divided by a model-error-inflated scale,
  so the headline "residual against the measurement noise alone" understates the real
  misfit by that factor. The README calls this "the number that says how well the forward
  model matches the organisers' processing".
- The NLL weights each curve by `1/(σ² + η²)`, so the az 135/225 geometries are
  down-weighted by up to 20×. Those are the α = 135° geometries — the longest shadows and
  the most shape information in the whole dataset.
- `NOISE_PROFILE` is built from these σ, so training injects noise with the wrong overall
  level and the wrong per-azimuth shape, teaching the flow to distrust the same geometries.
- The polish step, "stopped at the noise level", stops far too early.

**Suggested fix.** Estimate σ per curve from the high-frequency content (successive
differences, or the tail of the periodogram) rather than from the pair difference, and let
the A/B mismatch land in `eta`, where the log-determinant term already prices it correctly.
Keeping the pair difference as a *separate* diagnostic of A/B consistency is still useful.

## 2. The calibration is bounded by the optimiser budget, not converged

`calibrate.py` runs 150 Adam steps at lr 0.03. Adam's per-step magnitude is ≈ lr, so the
total movement of any raw parameter is bounded by 150 × 0.03 = 4.5:

```
rho   raw0 = -1.39  →  fitted 0.911 needs raw = +2.33   (moved 3.71 of 4.50, 82%)
                       reachable range from init: [0.003, 0.957]
eye   raw0 =  8.00  →  fitted 8.51  (moved 0.51)
                       reachable range from init: [3.53, 12.50]
```

ρ used 82% of its budget and stopped just under the reachable ceiling — it was still moving
when the run ended. `eye_distance` **cannot exceed 12.5** from an init of 8.0 whatever the
data says, so 8.51 is not evidence that the fit prefers 8.51; it is close to where it
started.

That matters, because 8.5 is probably too close. Two independent estimates:

- A 100 mm lens on full frame is ≈ 13.7° vertical field of view. Framing a body of
  half-height 1 model unit to fill most of the frame puts the camera at ~20 model units.
- The reference model, fitting a pinhole camera distance to the re-rendered Blender curves,
  lands at ~24 model units (broad optimum over 20–32, consistent across asteroids 1, 2 and
  3). Its pre-re-render fit was ~8, which is suspiciously close to this init.

**Suggested fix.** Raise `--steps` / `--lr` until the fit stops moving, and print the
movement of every parameter in raw space at the end so a truncated fit is visible. It is
worth re-checking ρ specifically: `Instrument.__init__` argues at length for starting at
0.20, and the data is pulling hard in the opposite direction.

## 3. There is no BRDF freedom, and the data asks for some

Regressing each real intensity curve on the corresponding Blender curve (both
mean-normalised, phase-aligned by circular cross-correlation), the amplitude ratio
`k = amplitude(real)/amplitude(blender)` falls monotonically with solar phase angle.
Model 3, grouped by α (recall α = arccos(cos ε cos az), so az 135 and az 225 are both
α = 135°):

```
α =   0°   (az 0)          k ≈ 1.17
α =  45°   (az 45, 315)    k ≈ 1.03
α =  90°   (az 90, 270)    k ≈ 0.88
α = 135°   (az 135, 225)   k ≈ 0.70
```

Real curves are 30% flatter than a Lambertian render at the largest phase angle. Two
mechanisms could do that — a rough-surface BRDF (Oren–Nayar-like), or interreflection
filling the shadows — and the chain currently has neither available:

- Scattering is pure Lambert. The OETF is a monotone map on *pixel values*, applied
  identically to every frame, so it cannot produce a phase-angle-dependent amplitude change.
- Interreflection can, through ρ — but ρ is railed against the optimiser budget (§2).

The calibration report agrees about where the problem is. Model 3 intensity `per_s`
(residual over `sqrt(σ² + η²)`, so *already* generously scaled per §1):

```
az    0:  1.22  1.25  0.63  0.66
az   45:  1.20  1.16  0.46  0.47
az   90:  0.31  0.39  3.02  1.14
az  135:  3.15  3.11  3.58  2.59      ← α = 135°
az  225:  2.63  2.82  3.26  2.13      ← α = 135°
az  270:  1.49  0.36  0.56  0.42
az  315:  0.43  0.32  0.29  0.41
```

A residual larger than the curve's own spread means those columns currently carry no usable
information. They are exactly the ones §1 also down-weights by up to 20×, so the two
findings compound: the geometries the model fits worst are the ones the likelihood has been
told to care about least.

Worth noting in passing that `forward/convex_egi.kernel` uses Lommel-Seeliger + Lambert
while the exact chain uses pure Lambert, so the repository's two forward models disagree
about the scattering law. That is defensible while the convex stage only supplies a starting
support, but it means the convex stage cannot be used as a check on the exact one.

**Suggested fix.** Give the chain one BRDF parameter (a roughness in an Oren–Nayar term is
the cheapest thing that produces the observed phase-angle trend) and fit it alongside ρ,
then re-read the az 135/225 residuals. If ρ alone can do it once §2 is lifted, that is the
simpler answer.

## 4. Recentring xy on the solid centroid is wrong and unnecessary

`shapes.rescale_touch_z(v, f)` and `CodeOperator.canonical` both translate the body so the
solid centroid sits on the rotation axis. The released STLs are *already* posed on that
axis, and recentring moves them off it. Posing each public STL to z ∈ [−1, 1] and measuring
the maximum xy radius:

```
              published R    STL origin           centroid-recentred
model 1          1.120       1.1198  (−0.0002)    1.1200  (−0.0000)
model 2          1.420       1.4142  (−0.0058)    1.4451  (+0.0251)
model 3          0.880       0.8782  (−0.0018)    0.8770  (−0.0030)
```

Left as they are, all three reproduce the published bounding-cylinder radius to within
0.02–0.6%, which is the check that the STL origin *is* the rotation axis. Recentring makes
model 2 four times worse and pushes it to 1.445 — larger than the published *minimal*
enclosing radius, which the true body cannot be. The displacement is up to 0.031 in xy.

This lands in two places that matter:

- `calibrate.py:load_truth` renders the displaced truth mesh, so every instrument parameter
  is fitted against a body that is slightly off-axis.
- `CodeOperator.physical` imposes it on every reconstruction iterate and then rescales xy to
  the published R, so the recovered body is both shifted and mis-scaled relative to a truth
  posed on the axis. For an asymmetric body the error grows with the asymmetry.

**Suggested fix.** Drop the xy translation; take the radius as max|xy| about the origin.
The lightcurves are nearly blind to a lateral offset, so this is a prior, not something the
data will correct — which is a reason to make it the right prior rather than a convenient
one.

## 5. The field of view is tied to the body, not to the lens

`fov = 2·atan(fov_scale · extent / eye_distance)` with `fov_scale = 1.6` fixed means the
body always fills the same fraction of the frame, and the modelled field of view moves with
the body instead of staying at the lens's:

```
R ≈ 1.2  body   →  fov_y ≈ 33°
model 10 (R = 3.95) →  fov_y ≈ 75°
real 100 mm lens on full frame  ≈ 14°
```

Two consequences. The cos⁴ falloff and the vignetting polynomial are applied at
body-relative radii rather than frame-relative ones, so they mean something different for
each body. And because `tan(fov/2) = fov_scale · extent / eye_distance`, the perspective
strength is `extent/eye_distance = tan(fov/2)/fov_scale` — fixing `fov_scale` couples the
frame fill to the camera distance, so the two cannot be identified separately. That coupling
is a plausible part of why §2 looks the way it does.

Most of the absolute scale cancels in the per-curve mean normalisation, so this is smaller
than it first appears — but model 10 is currently rendered through a wide-angle lens it was
never filmed with.

**Suggested fix.** Make the field of view a fitted instrument parameter (or fix it from the
lens and let the frame fill follow from `eye_distance`), rather than deriving it from each
body's extent.

## 6. Smaller things

- `data_io.resample_curves` decimates 841 → 48 (calibration) and 841 → 96 (reconstruction)
  with plain `np.interp` and no anti-aliasing. Features narrower than ~9 frames alias.
  Mostly harmless for smooth intensity curves; the binary curves of faceted bodies are the
  place it would show.
- The instrument is calibrated at 48 phases (`calibrate.py --phases`) and used at 96
  (`build_corpus.py`, `reconstruct_lpd.py`). Not a correctness bug — the chain is
  phase-count agnostic — but the fitted values have never been checked at the grid they run
  on.
- `forward/sdf_volumetric.py`, `forward/polytope_raycast.py` and `forward/sdf_surface.py`
  (660 lines) are not referenced anywhere in the package, the scripts or the tests.
- Every forward-model test is an internal-consistency test. Nothing compares the chain
  against the published curves except `calibrate.py` itself, which is also the thing being
  fitted. A regression test pinning the public-model residuals would catch a convention
  regression that the consistency tests cannot.
- `LitCoverage` relies on the mesh being closed: a face with `n·s < 0` is excluded only by
  losing the depth test to the front of the body, and there is no `(n·s) > 0` guard. True
  for everything FlexiCubes produces, but it would silently light back faces on an open mesh.

---

## One ambiguity both codebases share

The challenge text says the orientation pairs to match are 45↔225, 90↔270, 135↔315. That
pairing is geometrically impossible. The only rigid motion that flips a body upside down and
leaves the light along −x in the body frame is a 180° rotation about the x-axis; any
additional turntable rotation about z is absorbed into the "temporal shift" the text already
mentions. That motion maps camera azimuth β to −β, giving the pairs 45↔315, 90↔270,
135↔225 — agreeing with the published list on 90↔270 only.

Both codebases assume the bottom column of azimuth group θ is the geometry (θ, −e_θ), and
the reference model validated that against the Blender curves at corr 0.996 on asteroid 3,
so the *intended* geometry is almost certainly what both of us use, and the published
sentence is likely loose wording about which recordings were compared during curve matching.

The one thing still worth an experiment is the bottom camera's **elevation**. The elevation
table splits cleanly into 21° at az 0, 26° at az 45/90/135 and 24° at az 225/270/315, which
looks like two sessions with the tripod reset between them. If group 45's bottom column
comes from the session that was set at 24°, its elevation is −24° and not −26°. Both
codebases currently use −e_θ from the same group. Cheap to test: refit with the bottom
elevations taken from the paired azimuth instead and compare the per-geometry residuals.

---

## Reproducing the measurements

All measurements used the released `HAC_data_May_8` archive and needed only numpy plus this
package.

- Convention comparison: build `conventions.cameras()` vectors and `exact.rotate_z(w,
  −psi_grid())`, compare elementwise against the reference model's camera/light directions.
- §1: read `Asteroid0{m}_lightcurve_{intensity,binary}.txt`, take columns `4i` and `4i+1`
  per azimuth; compare `sqrt(mean((a−b)²)/2)` with `sqrt(mean(diff(a)²)/2)`, and the std of
  a 15-frame box smooth of `a−b` with the std of `a−b`.
- §2: `150 * 0.03 = 4.5`, then invert `softplus`/`sigmoid` at the init and fitted values
  quoted in `models/instrument_calibration.json`.
- §3: resample real and `_blender` curves to 360, mean-normalise, align each column by
  circular cross-correlation, and take the least-squares slope of `real − 1` on
  `blender − 1`.
- §4: `stl_io.load_stl` on each public STL, pose to z ∈ [−1, 1] with and without the xy
  centroid shift, compare `max|xy|` against `conventions.CYLINDER_R`.
- Otsu robustness: synthetic limb-darkened discs through `raster.otsu_threshold` at varying
  fill fraction, background level and background noise.
