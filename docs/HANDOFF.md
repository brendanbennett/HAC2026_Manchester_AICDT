# Handoff: the non-convexity work on `hac26` (Helsinki Asteroid Challenge 2026)

Written for whoever picks this up next. Branch `Jacks_shapes`. All changes are already written
into the working tree at `/home/m12567jp/dev/Helsinki` and are **uncommitted** — `git add -A &&
git commit` when you are satisfied.

Every number in this document was measured in this session unless marked *(earlier)*, meaning it
came from an earlier measurement round on the same codebase. Nothing here is estimated.

---

# 1. The one-sentence summary

The pipeline could not produce a non-convex body, for a reason nobody had found: the correction
field was **identically zero for every body ever fitted**, and its gradient was identically zero
too, so no amount of training could ever have recovered. That is fixed, the representation is
replaced, and decoded bodies now come out at convexity 0.62–0.82 where the convex core alone is
0.998.

---

# 2. What the shape representation is

```
f(y) = max_j (n_j . y - h_j)  +  Delta(y)
Delta(y) = sum_k g_k exp(-||(y - p_k)/sigma||^2 / 2)
h        = softplus( inv_softplus(h_base) + expand(dh) )
```

- `p_k`: a **fixed** 12x12x12 lattice of sites over `[-1.1, 1.1]^3`, cell centres. Never learned,
  never stored in a checkpoint. `LATTICE_SHAPE`, `LATTICE_EXTENT` in `hac26/field.py`.
- `sigma = 0.9 * spacing = 0.165`, per axis. `LATTICE_ALPHA`.
- `g`: 1728 signed amplitudes. **This is the code.** No latent, no decoder, no positions.
- `dh`: 128 values on `dir_design(128)`, band-limited to spherical-harmonic degree <= 5 by
  construction (`sh_expand` is `Y_dst @ pinv(Y_src)`, whose column space IS the degree <= 5 span).
- `CODE_DIM = N_DIR + N_SITES = 128 + 1728 = 1856`.

The predecessor was 32 cross-attention tokens, `32 x (p in R^3 + z in R^16) = 608`.

---

# 3. The four defects that were blocking non-convexity

## 3.1 `TokenField` produced Delta = 0, always

`hac26/field.py` initialised `p` and `z` to zeros. Identical tokens make every `k(z_i)` and every
`||y - p_i||` identical, so the softmax is exactly uniform, `a @ v(z)` is constant in `y`, and
`out - out.mean()` is **identically zero**. The gradient is exactly zero as well, for two
independent reasons: the softmax weights sum to one so `sum_i d a_i/d p_j = d(1)/d p_j = 0`, and
a `z`-perturbation is `y`-independent so the mean subtraction removes it.

Measured *(earlier)*: 400 Adam steps moved the loss 0.531236 -> 0.531236, `max|Delta| = 1.8e-7`;
the committed results show posterior `spread` 0.9932 / 0.9941 / 0.9914 — the draws were 99%
identical because they were all exactly the convex core.

Fixing the init alone (before the representation was replaced) was worth **9-45x on fit loss**,
**+0.026 to +0.055 Dice**, and took a bilobe from neck ratio 1.02 ("no waist at all") to within
**2-8% of the true waist** with only 32 tokens.

## 3.2 The softmax could not deepen a carve

Structural, and the reason the token field was replaced rather than repaired: the softmax weights
**sum to one**, so co-located tokens *average* rather than add. You cannot carve deeper by
clustering, and `sigma` is not a reach — it is the softness of a Voronoi partition of all space.

Measured *(earlier)*: a surface shell saturates at **0.36 R** of carve depth, identical to four
decimal places from N=513 to N=4095 (its numerical rank saturates at 838). A volumetric lattice
reaches **0.64 R**. At the body centre a shell's reachability is 0.47 against a lattice's 1690.

## 3.3 `Delta` depended on the query batch

`TokenField.forward` ended `out - out.mean()` — the mean over *the call's own points*. Invisible
while Delta was zero. Live, `extract_mesh` evaluates the grid in chunks of 262144, so each chunk
had a different constant subtracted: measured a **0.0209 step in the SDF at a chunk boundary, 0.84
of a grid cell at res=128**, with a boundary landing at x = -0.002 (the centre of the body) at
res=80. The fit (zero-mean over ~9000 sample points) and the extraction (over ~274k grid points)
also disagreed by up to **2.1% of R** — a bias no training can remove, because the two measures
differ by construction. A fixed-site sum is batch-independent; there is no mean subtraction now.

## 3.4 The reconstruction ran in the wrong frame

**This one is not new and was not mine — it was masked by Delta = 0, and it is larger than the
init bug.** The corpus is fitted at `ImplicitBody(radius=1.0)` on bodies posed to xy `r_max = 1`,
and the operator is trained at `curves_from_code(code, 1.0, ...)`. `reconstruct_lpd` decoded at
the **published R** (0.67 ... 3.95). Token positions came straight from the code unscaled while
`sigma = 0.25R` and `s = 0.15R` scaled with R. The challenge pose is **anisotropic** — z is always
[-1,1] and only xy scales — so no single scalar can be right in both.

Same fitted code, same truth mesh, decoded both ways, Dice against the physical truth:

| published R | convex core (floor) | decode at R (the old code) | decode canonical, then `fit_to_cylinder` |
|---|---|---|---|
| 0.67 (model 9) | 0.9311 | **0.8478** — worse than no correction at all | **0.9705** |
| 1.42 (model 2) | 0.9285 | 0.9322 (+0.004) | **0.9700** (+0.042) |
| 3.95 (model 10) | 0.9306 | 0.9341 (+0.004) | **0.9698** (+0.039) |

The canonical path is flat at 0.970 at every R, which is what correctness looks like. The fix is
the repo's **own written convention**: `hac26/shapes.py::canonicalize_r` states verbatim
`train target: canonicalize_r(hull)` / `test output: fit_to_cylinder(prediction, R)`, and
`scripts/eval_exact.py` already did it for the convex path. `reconstruct_lpd` was the only place
that did not.

Side benefit: `extract_mesh` uses a **cubic** grid of half-width `1.6 * radius`. At R = 3.95 that
gave the body **7.6 cells across its z extent**; canonical gives 40 at the same `--res 64`.

---

# 4. Change log by file

## `hac26/field.py`
- `TokenField` -> `GaussianLattice`. `g` is the only parameter; `p`, `inv2`, `pb` are
  `persistent=False` buffers so a checkpoint cannot redefine another body's lattice.
- `forward` uses the **expanded square** so the cross term is one matmul:
  `||(y-p)/s||^2 = sum y^2/s^2 + sum p^2/s^2 - 2 (y/s^2).p`. Measured on 2 CPU cores at a 64^3
  grid: broadcast form 14,573 ms, matmul form **3,773 ms**, agreeing to 1.7e-05 (float32 noise).
- `LATTICE_CHUNK_ELEMS = 6e6`, raised 16x on CUDA. Chunk size is **non-monotone**: at res=64, 6e6
  beats 6e7 by **3.3x** on CPU (cache-bound) while a GPU wants the opposite.
- `CORE_SCALE` **deleted**. Algebraically redundant (`s * sum g_k E_k == sum (s g_k) E_k`; fits at
  `s=0.15R` and `s=1` agree to 8.7e-11) but it silently set the units `g` learns in, hence the
  effective learning rate and the scale the flow must whiten away.
- `ImplicitBody` gains `dh`, `dh_expand` (non-persistent), `support()`, `set_support()`.
  `ConvexCore.forward` gains an optional `h=` override so the dh-corrected support stays
  differentiable without ConvexCore knowing dh exists.
- `_real_sh`, `sh_expand`, `dir_design` — the band-limit machinery.
- `spherical_design` caches atomically, refuses n > 512 (measured **13.8 s and 3.6 GB per
  iteration at n=4096 -> ~15 h** for the default 4000), and **only caches at the default
  `iters`** — a 20-iteration design has residual 7.7e-04 against 4.8e-07, 1600x worse, with an
  identical `design_sha`, so nothing downstream could have detected the poisoning.
- `design_sha` digests at **float32**, the precision `ConvexCore` stores normals at.
- `extract_mesh` asserts the extent covers `LATTICE_EXTENT + 3 sigma`.

## `hac26/solvers/lpd_flow.py`
- `PrimalNet` is now two structured branches plus shared conditioning, not a flattening MLP:
  - `VolBranch` — 3-D CNN over the 12^3 lattice, zero padding (the box is not periodic), no
    culling (it would break the fixed index set). 6 input channels: `g_t`, `core_sdf` at the
    sites, the inside indicator, and normalised x, y, z. **`core_sdf` is what replaces culling.**
  - `SphereBranch` — `SphereConv` from five fixed operators (I, an 8-NN mean, two tangential
    first moments, one second moment). `S_0 = I` gives the identity path for free, which is why
    the old `hidden > CODE_DIM` rule disappears with the flattening.
- **Parameter count: 2,367,348, down from 2,681,345**, while the code went 608 -> 1856. A flat MLP
  obeying the old rule would have been 13.1 M.
- `_summaries` -> `_summary`: pooled over **geometries only**, then a shared per-mode
  `Linear(96,16)`. The old version summed over geometries AND modes while dividing by the geometry
  count alone — **exactly N_MODES = 40 times a true mean**, measured 40.000004, mask-independent.
  Verified with a stub: a constant 3.5 in gives 3.5 out; the old code gave 140.0.
- Padded mode slots are masked. At `--phases 16` they contributed **81.7% more norm than the real
  modes** — a large data-independent constant. Detected from the input, not a constructor
  argument, so a checkpoint cannot disagree with a run.
- `r_perp` **deleted** everywhere. It was `r_perp[..., :4] = 0`, a hard cut of rotation orders
  1-4, which the docstring called "the projector Pi onto its range" — no projector was ever
  computed. It cost a full extra dual forward pass and 2 of its 6 channels were never assigned.
  -98,304 params.
- `time_embed` — a **separate** embedding for t. Reusing `fourier_embed` (scaled for m = 1..40)
  gave the six step times singular values `[6.85, 1.03, 7.7e-2, 3.2e-3, 7.2e-5, 1.0e-6]`,
  **condition number 6.7e6**, effectively rank 4 — and the gain path, whose whole job is to learn
  a function of t, was reading that.
- `CodeCodec` — **per-block scalar** whitening with a median-absolute-deviation scale, plus
  `asinh(g/g_s)`. Round-trip exact to 3.7e-08.

## `scripts/fit_shapes.py`
- `BatchedFit` — all bodies' `h` and `g` as two stacked tensors, one batched matmul per step
  instead of a Python loop of B small ones. **Verified arithmetically identical to the per-body
  loop: relative difference 0.000e+00.** This could not have been written before; deleting the
  shared decoder is what made the bodies independent.
- **THE ENCODER IS SEQUENTIAL.** `h` is pinned to the analytic support of the body's own convex
  hull and never moves; only `g` is fitted. See §6.
- The shared decoder, `--decoder`, and `runs/token_decoder.pt` are gone entirely.
- SDF supervision extended to cover the lattice plus 3 sigma; sites outside the sampled domain
  would otherwise be unconstrained and free at sampling time.
- `report_corpus` runs **after** the corpus is written, never before — a fit that took hours must
  be flagged, not thrown away.
- TF32 on CUDA.

## `scripts/train_lpd.py`
- `flow_loss` rewritten. **The operator is applied at `x1_hat = x_t + (1-t) v0(x_t)`, not at
  `x_t`** — `v0` is a first pass with the residual features zeroed, costing one network forward
  and **no operator call**. This matters: `x_t` at small t is mostly `x0`, and a decoded Gaussian
  tail draw had `|g|` up to 3.77 against a corpus max of 0.226.
- **dh supervision by perturbation.** A corpus body's `h` is exact so its dh block is zero. `eps`
  is redrawn per draw, the operator runs at `softplus(inv_softplus(h) + expand(eps))`, and the
  flow's target for that block is `-eps`. No new pipeline stage, no extra operator call, and it
  doubles as free augmentation.
- Continuous `t`, stratified across the batch — `O(1/B^2)` instead of `O(1/B)`. One operator call
  per draw either way, so it is free.
- Per-block loss weight `1/N_block`. Unweighted, `g`'s 1728 dimensions take ~89% of the gradient.
- EMA with **zero-init and bias correction** (the two conventions are mutually exclusive; mixing
  them divides by 0.002 at n=2 and produces NaN — this happened and was caught). Window tied to
  the run: `decay = min(--ema, 1 - 10/steps)`.
- `--time-budget` (default 20 h): times three steps, projects the finish, and prints the largest
  `--steps` that would fit. It **warns rather than exits** -- the run checkpoints every
  `--ckpt-every` steps and resumes from the optimiser state, the RNG state and the
  early-stopping counters, so an overrun costs a restart, not the work.
- `support_residual_channel` — `A^T r` on the dh directions. See §8 for what it is not.
- Digest gates on both designs; `code_dim` in every cache key and meta.
- `fit_body` and `flow_targets` deleted (zero callers each).

## `scripts/reconstruct_lpd.py`
- Canonical frame throughout, `fit_to_cylinder(v, R)` at the end. `support_from_convex`
  canonicalises the vertices before taking the support — support is a max over vertices and does
  **not** transform under an anisotropic scale by any scalar.
- Collapse warning gated on the **off-medoid** mean Dice, which is sample-count independent
  (`spread` includes the medoid's Dice against itself: at 2 draws `spread > 0.95` needs 0.900, at
  16 it needs 0.947).

## `hac26/shape_library.py`
- New `contact_binary` base class: two ellipsoids with centre separation forced to
  `>= 0.90 (a1x + a2x)` and the fillet capped at 0.01-0.05, weight 0.16.
- `base_kind` drawn **once**, outside the non-convexity retry loop. Redrawing it per attempt
  biased the realised distribution towards whatever survived the gate first — prism 0.12 ->
  **0.037**, polytope 0.12 -> **0.225** — while `write_report` printed `base_weights` as if it had
  been honoured.

---

# 5. Non-convexity: what actually improved

Decoding fitted corpus codes, sequential encoder, res=48:

| body | decoded | core alone (code zeroed) |
|---|---|---|
| 0 | **0.6745** | 0.9988 |
| 1 | **0.6248** | 0.9985 |
| 2 | **0.8230** | 0.9977 |
| 3 | **0.7170** | 0.9969 |

Under the old field every one of these was the core-only value. They now sit inside the corpus's
own convexity band (median 0.655 *(earlier)*). `fit_to_cylinder` restores `r_max = 3.950` exactly.

Corroborating: `max|Delta| = 0.92`, `|g|max = 0.64`, amplitude variance 0.00112.

---

# 6. The sequential encoder — read this before changing the fit

`h` and `g` overlap. Any body can be written as a larger core carved more deeply or a smaller core
carved less, and the two agree exactly in the low spherical-harmonic degrees — principal cosines
**1.00000 at l=0**, 0.996 for the three l=1 translations, 0.987-0.993 at l=2 *(earlier)*. Solved
jointly, the same body at the same accuracy admits a family of `(h, g)` pairs, and a flow trained
on that family faithfully learns the ambiguity as spurious multimodality.

There is a second reason, specific to this pipeline and worse: **at reconstruction `h` does not
come from the fit at all.** It comes from `support_from_convex()`, the convex stage's estimate of
the body's hull. If the corpus's `h` were free to drift, every corpus `g` would have been fitted
against a core meaning something different from the core it is decoded against.

Measured: joint fitting drifts `h` by **0.3517** in units where `h` is order 1 — a third of the
support, silently. The cost of freezing is convergence rate, not accuracy: joint reached 0.000231
by step 250 and frozen was still at 0.000446, but frozen passes it by **step 800 (0.000089)** with
only `g` to fit. `|g|` is larger, as it must be — the core is now the full hull.

**`FIT_STEPS` should not be reduced below ~1000 because of this.**

---

# 7. Convexity-bias audit — every candidate, measured

| candidate | measured | verdict |
|---|---|---|
| extraction resolution 24 -> 64 | convexity 0.6749 -> 0.6676 | 0.007, negligible |
| mesh decimation to 600 faces (what the surrogate sees) | 0.6676 -> 0.6699 | +-0.002, neutral |
| planar snap, 12 planes | `RECON_SNAP=0` | off by default; would be a bias if enabled |
| degenerate mesh -> zeroed prediction path | 8/8 pure-noise draws decoded to usable meshes | rare |
| convex core carrying concavity information | a bounding spheroid with none gives identical feature accuracy to the true hull *(earlier)* | supplies scale only |
| EMA weight averaging | averages over training steps, not over the posterior; `x0` is still drawn fresh per sample | not a posterior collapse |

**The real convexity pressure is not any of those.** It is that an under-trained velocity
approximates `E[x1 - x0 | x_t, cond]`, which transports every draw toward the corpus **mean** code,
and a mean over bodies is smoother than any of them. This is inherent to flow matching early in
training. It is now instrumented — every validation prints:

```
step 1  |g|hat 0.01563 vs corpus 0.01582 (99%), across-draw spread 0.01500
```

mean `|g|` of the model's own endpoint estimate against the corpus, and the spread across draws.
**A collapse drops the first and kills the second. Watch this number.** At init it reads 99% with
a spread comparable to the magnitude — that is what "no collapse" looks like.

---

# 8. Current limitations, ranked by how much they should worry you

1. **`sigma = 0.165` caps the finest expressible feature.** Faithful down to `waist/R ~ 0.20`
   *(earlier)*; below that the correction smooths. Neck error does not saturate with N — log-log
   slope ~ -0.31 — but at N = 8000 a severe waist is still **+48% too fat**, and +10% would need
   ~4e6 sites. **More sites is not the route to a sharper waist.**
2. **The `dh` supervision is entirely synthetic.** `DH_EPS_STD = 0.02` was taken from the measured
   `dh` a fitted body needs (0.0199 R), **not** from the actual error of your convex stage. If
   `h_conv` is off by more than the perturbation distribution covers, the flow has never seen the
   correction it is being asked to make. **Measure `|mesh_support(truth) - support_from_convex()|`
   on the three public bodies and set `DH_EPS_STD` from it.** This is the highest-value open item.
3. **The adjoint channel is not the adjoint.** `support_residual_channel` is `A^T r` on the dh
   directions. It does **not** include the normalisation Jacobian `DN`, and it does **not** include
   `J = d(areas)/dh`. Both need the polytope rebuilt per sample; `J` alone is 4096x4096 per body.
   It localises (single-camera residual -> peak/mean 1.31) but it is a hint, not a gradient. Note
   `A_conv` is provably blind to concavity anyway: a notched cube's unshadowed curves are
   reproduced **exactly, as a fatter box**, deep pits are 3x under-signalled, and 90% of its
   row-space power sits at l <= 3 *(earlier)*. **Never add it to the velocity.**
4. **The band limit is exact on the softplus argument, ~95% on `h`.** `d/dx softplus = sigmoid` is
   not constant and `h` varies across normals, so the induced change in `h` carries ~5%
   out-of-band content. First-order — it does not shrink with the perturbation. Far below the
   fully-white dh that kills 7% of facets at N=128, but not zero.
5. **The corpus's deep-neck tail is thin.** `contact_binary` was added and produces convexity
   0.750 on a spot check, but the library's neck distribution has **not** been re-measured at
   scale since. Do that before trusting the corpus for model 3.
6. **The medoid prefers the most central draw.** A mild preference for the less-committed body.
   Inherent to the scoring strategy, not a bug, but it is a convexity-ward selection.
7. **`lpd_flow.py` has no committed tests.** By deliberate instruction — only single-use checks
   were run. The verifications in §9 exist only in this document.
8. **`--time-budget` is a projection, not a scheduler.** It warns once, from three measured
   steps; it will not notice a run that slows down later.
9. **The reconstruction codes are persisted** to `<out>.codes.npz` (raw codes, the support they
   were decoded against, the radius) *before* any draw is decoded, so a degenerate-draw exit or a
   crash in the medoid still leaves them. They are also the only record of the posterior -- the
   STL keeps the medoid alone, so without them a convex result cannot be told apart from a medoid
   that picked badly. Re-extracting at a different `--res` or medoid rule from a saved file costs
   minutes against the ~24% of the operator budget a fresh reconstruction costs.

10. **All flow-training numbers in this session are meaningless as accuracy.** They were produced
   with an untrained surrogate over 4-6 bodies and 4 steps. Only the *shapes*, the plumbing and
   the diagnostics were being tested.

---

# 9. How to check the work

```
python -m pytest tests/ -q -m "not slow"          # 40 passed, 6 skipped, 3 deselected
python scripts/fit_shapes.py --bodies 600 --shapes-dir <lib> --steps 4000 --out runs/corpus_codes.npz
```
`fit_shapes` must print `[check] max|Delta| ...` well above zero and a non-trivial `g variance`.
If `max|Delta| < 1e-4` the correction is dead and the corpus is refused (it is still written, so
the compute is not lost).

Then `train_lpd`, watching two lines: the `[budget]` projection in the first minutes, and the
`|g|hat ... vs corpus` tripwire at every validation.

---

# 10. Traps

- **Two designs, both load-bearing.** `hac26/design4096.npy` (the core's normals, present, 98,432
  B) and `hac26/design128.npy` (the dh directions, newly committed, residual 2.3e-17). Both are
  digest-gated into the corpus meta as `design_sha` and `dir_sha`. `h` and `dh` are indexed **by
  direction**; two machines generating either independently would disagree about what every
  coefficient means, silently.
- **`scripts/_venv_setup.sh` exists** (7,851 B). An earlier note in this session doubted it; that
  doubt was wrong.
- `run_remote_pipeline.sh` references `$DECODER_FILE` in three `stage_signature` printfs at lines
  *before* its old assignment; under `set -u` that survived only by evaluation order. All of it is
  removed together — do not reintroduce one half.
- `ablate_flow.py`'s loss must change in lockstep with `train_lpd.py`'s or the ablation stops
  being comparable to training, which is the only thing it exists for.
- `ablate_flow.py` used to claim the target is `(x1 - x_t)/(1 - t)` so late draws "carry a gain of
  6". In this parameterisation that expression is **identically `x1 - x0`** — the target does not
  depend on t. Do not build a per-t weighting on that.
- `N_BODIES` stays at **1000**. It was briefly lowered to 600 on the premise that nothing
  survives the 24 h deadline, so the corpus build -- 39% of the operator budget by camera-passes
  -- would be repaid every run. That premise is false: only the GPU processes are killed, the
  filesystem persists, `save_corpus_cache` copies the corpus to `runs/` from a trap on
  EXIT/INT/TERM (CRC-checking the archive first), and the next run seeds `/tmp` from it. The
  corpus is built once. `train_lpd` likewise checkpoints and resumes, and the corpus build itself
  resumes per body through `{cache}.parts/`.

---

# 11. Speed, and what is left

Done: the vectorised fit (bit-identical), the matmul-form lattice (14,573 -> 3,773 ms), TF32 on
all three entry points, `r_perp`'s deletion halving the dual work, and moving `fit_shapes` to the
GPU at all — it had **no `.to(device)` anywhere**, which was survivable at 32 tokens and is not at
1728 sites (~1.7e12 element-ops at the remote defaults).

**Considered and rejected: batching the field evaluation inside `extract_mesh` across the flow
batch.** Profiled at the training resolution: `extract_mesh` at res=32 is **49% field evaluation,
51% FlexiCubes and overhead**, and FlexiCubes is a topology pass that cannot batch. At the default
`FLOW_BATCH=2` the ceiling is therefore ~1.3x on `extract_mesh` and well under that on the whole
operator call, against a change that touches the extraction path every script uses. Not worth it
at B=2. It becomes worth reconsidering if `FLOW_BATCH` rises substantially.
