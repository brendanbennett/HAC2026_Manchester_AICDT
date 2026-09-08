# Forward-model review, cross-referenced against Helsinki-Challenge-2026

Review of `hac26/forward/` and the calibration that fits it, done by cross-referencing an
independently developed forward model for the same challenge
(`Helsinki-Challenge-2026`, a Lambertian mesh renderer validated against the published
Blender curves). The two codebases share nothing but the challenge description, so
agreement between them is evidence and disagreement is a place to look.

Everything below is either a numerical comparison between the two models or a measurement
on the released data (`HAC_data_May_8`). Reproduction notes are at the end.

## Status

| | finding | status |
|---|---|---|
| 1 | the pair difference is not the noise | **fixed** -- `noise.sigma_from_highfreq` |
| 2 | the calibration is bounded by its step budget | **fixed** -- early stop, movement report |
| 3 | no BRDF freedom, and the data asks for one | **open** -- the evidence stands, the mechanism I proposed does not; see the note added below |
| 4 | the xy centroid recentring | **fixed** -- `rescale_touch_z(..., centre_xy=False)` |
| 5 | the field of view is tied to the body | **open, and smaller than first stated** -- see the correction below |

Findings 1, 2 and 4 are commits on this branch, each with tests. Nothing here has been rerun
through `calibrate.py`: that needs nvdiffrast and a GPU, and 1, 2 and 4 all change what the
calibration means, so `models/instrument_calibration.pt` is stale and everything downstream
of it should be regarded as provisional until it is refitted.

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

## 1. The replicate-pair difference is not measuring measurement noise

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

Measured on the released curves — the pair difference against the noise estimated from each
curve's own high-frequency content, plus how much of the difference survives a 15-frame box
smooth:

```
                    pair difference / noise, per azimuth
model 1 intensity   az0  4x  az45  3x  az90  3x  az135 12x  az225 50x  az270 11x  az315  7x
model 2 intensity   az0 11x  az45 24x  az90 39x  az135 81x  az225 71x  az270 42x  az315 26x
model 3 intensity   az0  6x  az45  3x  az90  3x  az135  5x  az225  6x  az270  2x  az315  7x

over all 168 released curves:    min 1x, median 12x, max 277x
low-frequency share of (a − b): 0.86 to 0.99
```

σ is inflated by a median factor of 12, and 86–99% of the difference is low-frequency —
structure, not noise. Four consequences, all downstream of the same number:

- The `per_sigma` column of the residual report is divided by a model-error-inflated scale,
  so the headline "residual against the measurement noise alone" understates the real
  misfit by that factor. The README calls this "the number that says how well the forward
  model matches the organisers' processing".
- The NLL weights each curve by `1/(σ² + η²)`, so the az 135/225 geometries are
  down-weighted by up to two orders of magnitude. Those are the α = 135° geometries — the longest shadows and
  the most shape information in the whole dataset.
- `NOISE_PROFILE` is built from these σ, so training injects noise with the wrong overall
  level and the wrong per-azimuth shape, teaching the flow to distrust the same geometries.
- The polish step, "stopped at the noise level", stops far too early.

**Fixed.** σ now comes from the second difference along each curve,
`1.4826 · MAD(d) / √6`, taken at the files' native ~841 frames. Second differences rather
than first: the first difference still carries the signal's own slope, which on these curves
is of the same order as the noise and larger than it on the faceted bodies. The estimate has
to be made before the resampling to the operator's phase grid, so `load_model_curves` now
keeps the native-resolution curves. The pair difference survives as `noise.ab_mismatch`,
documented as the A/B consistency diagnostic it is and reported beside σ by `calibrate.py`;
`eta` is where it belongs, and the log-determinant term already prices it. `NOISE_PROFILE`,
`NOISE_LO` and `NOISE_HI` are re-measured with the new estimator — the profile now rises
monotonically with phase angle and the intensity and binary curves agree on its shape, which
the old one did not.

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

**Fixed.** The run now stops early on a plateau of the likelihood (`--tol` over
`--patience` steps), so the step cap can be raised without paying for it when it is not
needed, and the default cap goes 150 → 600. Afterwards it prints how far every parameter
travelled against its own budget and names the ones still moving, and the residual table is
explicitly downstream of that check: while anything is named, the residuals are those of a
truncated fit. The movement table goes into `instrument_calibration.json` as well.

Worth watching ρ specifically on the refit: `Instrument.__init__` argues at length for
starting at 0.20, and the data was pulling hard in the opposite direction when the run ended.

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

**Open. The evidence above stands; the mechanism I first proposed does not.** The obvious
candidate is a surface roughness — an Oren–Nayar term on the direct light, zero being exactly
Lambert. I implemented it (per-face emission directions, the factor applied to the direct
term only, with the interreflected term left Lambertian) and measured what it does, and it
does not do this:

```
                        amplitude ratio vs a Lambert render, by phase angle
target from the data      α=0: 1.17   α=45: 1.03   α=90: 0.88   α=135: 0.70
Oren–Nayar, σ = 10°       α=0: 0.94   α=45: 0.98   α=90: 1.00   α=135: 1.00
Oren–Nayar, σ = 20°       α=0: 0.86   α=45: 0.96   α=90: 1.00   α=135: 1.00
Oren–Nayar, σ = 30°       α=0: 0.80   α=45: 0.94   α=90: 1.00   α=135: 1.00
```

It bites hardest at α = 0, where the data wants *more* amplitude, and does nothing at all at
α = 135, where the data wants a third less. Wrong sign at one end and no effect at the other.
An additive ambient floor — the lab is not a black void, and there is a beam splitter in the
path at az 0 — moves things the same negligible amount on the same test.

Two caveats on that negative result, which is why this is open rather than closed:

- The test body is a convex ellipsoid, because that is what the pure-torch rasteriser can
  render here in reasonable time. A convex body's high-phase-angle amplitude comes from the
  shape of its terminator, and both mechanisms are weak there. The trend was *measured* on
  model 3, which is strongly non-convex. The probe may simply be insensitive rather than the
  mechanism wrong.
- The qualitative Oren–Nayar model diverges as both the incidence and emission angles go to
  grazing, and unbounded it reaches a factor of ~3600 on a sphere at 20° of roughness, which
  saturates the sensor and destroys the curve instead of shaping it. Any implementation needs
  a floor on `max(n·s, n·v)`; 0.25 is a reasonable one.

So the branch does **not** add a BRDF parameter. Adding a fitted physical parameter to the
calibration on the strength of a hypothesis whose one usable test contradicts it is the same
mistake as §2 in a different costume — it would give the fit a new direction to absorb misfit
along, with no evidence it is the right one. The right next step is to repeat the measurement
above on a non-convex body with nvdiffrast, where the render is cheap, before deciding.

The other thing to try first is simply lifting §2 and seeing how far ρ goes on its own:
interreflection fills shadows, which is the right kind of effect, and ρ was still climbing
when the old run ended.

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

**Fixed.** `rescale_touch_z` takes a `centre_xy` flag, still true by default because a
procedural library body has no meaningful origin and has to be mounted somehow — the same
rule `shape_library.pose` already uses. The call sites handling bodies already in the
challenge frame pass `False`: the calibration's truth, both scorers, `reconstruct_lpd`'s dice
check, `measure_public_shapes`. `CodeOperator.canonical` no longer translates at all, so the
radius is `max|xy|` about the axis; a corpus body, which is centred when it is posed, is
unaffected. The published-radius test tightens from 3% to 1% and gains a companion showing
that centring makes the fit worse wherever the centroid really is off the axis.

The lightcurves are nearly blind to a lateral offset, so this is a prior rather than
something the data will correct — which is the reason to make it the right prior.

## 5. The field of view is tied to the body, not to the lens

`fov = 2·atan(fov_scale · extent / eye_distance)` with `fov_scale = 1.6` fixed means the
body always fills the same fraction of the frame, and the modelled field of view moves with
the body instead of staying at the lens's:

```
R ≈ 1.2  body   →  fov_y ≈ 33°
model 10 (R = 3.95) →  fov_y ≈ 75°
real 100 mm lens on full frame  ≈ 14°
```

**Correction to my first draft of this section.** I wrote that fixing `fov_scale` couples
the frame fill to the camera distance so that the two cannot be identified separately, and
that this was a plausible cause of §2. That is wrong. Perspective strength is
`extent / eye_distance`, which `eye_distance` sets on its own; `fov_scale` only fixes how
much of the frame the body fills. The two are separable and §2 stands on its own.

What is left is smaller and second-order: the cos⁴ falloff and the vignetting polynomial are
applied at body-relative radii rather than frame-relative ones, so a single fitted vignette
polynomial means something different for each body, and model 10 is rendered through a
75° lens it was never filmed with. Most of the absolute scale cancels in the per-curve mean
normalisation. Tying the framing to the body also has a real virtue — every body is sampled
by the same number of pixels, whatever its shape — so this is a trade, not a defect.

**Open, low priority.** If it is worth doing, make the field of view a fitted instrument
parameter and let the frame fill follow, rather than deriving it from each body's extent.

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
  per azimuth; compare `noise.ab_mismatch` with `noise.sigma_from_highfreq`, and the std of
  a 15-frame box smooth of `a−b` with the std of `a−b`. Both on the native-resolution
  curves — resample first and the noise estimate reads the signal's curvature instead.
- §2: `150 * 0.03 = 4.5`, then invert `softplus`/`sigmoid` at the init and fitted values
  quoted in `models/instrument_calibration.json`.
- §3: resample real and `_blender` curves to 360, mean-normalise, align each column by
  circular cross-correlation, and take the least-squares slope of `real − 1` on
  `blender − 1`.
- §4: `stl_io.load_stl` on each public STL, pose to z ∈ [−1, 1] with and without the xy
  centroid shift, compare `max|xy|` against `conventions.CYLINDER_R`.
- Otsu robustness: synthetic limb-darkened discs through `raster.otsu_threshold` at varying
  fill fraction, background level and background noise.
- §3: render an ellipsoid hull at geometries 0, 4, 8, 12 (α = 0, 45, 90, 135) with
  `Instrument(rho=0.85, tau_i=1e-4, quantise=False)` and the saturation pushed well clear, so
  no geometry sits on the intensity threshold; take the peak-to-peak of each mean-normalised
  intensity curve and divide by the same at zero roughness. Check the raw minima are non-zero
  first: with the default instrument the α = 135 curve sits on `tau_i` and collapses, which
  makes the ratio meaningless rather than small.
