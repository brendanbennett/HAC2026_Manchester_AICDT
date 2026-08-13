"""Training of the LPD on simulated shapes (only 3 public models exist, so the
training distribution is synthetic; see shapes.sample_training_shape).

Optimizer settings follow arXiv:1707.06474: ADAM with beta2 = 0.99, cosine-annealed
learning rate from 1e-3, global gradient-norm clipping at 1. Loss = N * MSE on the
scale-free EGI direction p, plus a closure penalty |sum_i p_i u_i|^2 (the Minkowski
feasibility defect).

Augmentations model the documented lab unidealities: additive noise on raw curves
before normalization, small independent per-curve cyclic shifts (residual alignment
error), and random curve dropout (missing files at higher difficulty levels).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from hac26.forward.convex_egi import ConvexPhotometricOperator, stack_A
from .geometry import build_cameras, make_grid
from hac26.solvers.lpd_convex import LPDNet
from .radial import (fibonacci_sphere, mesh_radial, support_ray_matrix,
                     torch_dice_loss)
from .shapes import (canonicalize_r, hull_mesh, mesh_support, mesh_to_egi,
                     sample_training_shape)


from .noise import NOISE_PROFILE, apply_noise


@dataclass
class Preset:
    name: str = "gpu"
    steps: int = 100_000
    batch: int = 16
    ch: int = 48
    n_iter: int = 15
    n_primal: int = 7
    n_dual: int = 7
    m: int = 360
    n_theta: int = 24
    n_phi: int = 48
    lr: float = 1e-3
    c_lambert: float = 0.1  # weakly identified on Blender curves; refit on real data
    sigma: float = -1.0     # FITTED on public models 1-3 (see data/conventions.json)
    delta: float = 1.0      # FITTED on public models 1-3 (see data/conventions.json)
    eps_norm: float = 1e-3
    # None = the per-camera heteroscedastic profile from hac26.noise;
    # "flat" = homoscedastic, i.e. what a noiseless-generation pipeline effectively assumes.
    noise_profile_mode: str = "measured"
    noise_lo: float = 0.005
    noise_hi: float = 0.03
    shift_max: int = 3
    drop_p: float = 0.15
    closure_weight: float = 1.0
    num_workers: int = 4
    ckpt_every: int = 1000
    log_every: int = 50
    seed: int = 0
    amp: bool = True
    amp_dtype: str = "bf16"   # "bf16" | "fp16"; see the note in train()
    support_head: bool = False   # predict h(u) instead of scoring only the EGI
    canonical_r: bool = False    # train on the r_max=1 canonical shape (see shapes.canonicalize_r)
    r_cond: bool = False         # feed the bounding radius R to the network as an input
    r_jitter: float = 0.05       # lognormal jitter on R during training, so a slightly
                                 # mis-specified R at test time does not derail the model
    egi_weight: float = 1.0      # weight on the EGI objective
    dice_weight: float = 0.0     # weight on the EXACT Dice metric (hac26.radial); this is
                                 # the scoring function itself, not a surrogate
    h_mse_weight: float = 1.0    # weight on the support MSE (kept small but non-zero when
                                 # training on Dice: it anchors the scale, which Dice --
                                 # being a ratio -- is completely blind to)
    n_rays: int = 1024           # sphere quadrature for the Dice loss
    dice_chunk: int = 256        # ray-axis chunking, keeps (B,V,N) off the GPU at once
    p_flat: float = 0.0          # fraction of flat-faced / few-face training bodies
    gate_rank: int = 0           # rank of the occlusion gate (0 = off); see lpd.LPDNet
    gate_bias: float = 3.0       # gate CNN output bias at init; see lpd.LPDNet


PRESETS = {
    "smoke": Preset(name="smoke", steps=200, batch=2, ch=32, n_iter=8,
                    n_primal=5, n_dual=5, m=120, n_theta=12, n_phi=24,
                    num_workers=0, ckpt_every=100, amp=False),
    "laptop": Preset(name="laptop", steps=5_000, batch=8, ch=32, n_iter=10,
                     n_primal=5, n_dual=5, num_workers=2, amp=False),
    "gpu": Preset(name="gpu"),
}


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SyntheticCurves(IterableDataset):
    """Streams (d, mask, p) triples; A is shared read-only numpy."""

    def __init__(self, A: np.ndarray, grid, pr: Preset):
        self.A, self.grid, self.pr = A, grid, pr
        self.rays = fibonacci_sphere(pr.n_rays) if pr.dice_weight else None

    def __iter__(self):
        wi = get_worker_info()
        seed = self.pr.seed + (wi.id + 1) * 10_007 if wi else self.pr.seed
        rng = np.random.default_rng(seed)
        C, m, _ = self.A.shape
        while True:
            s = sample_training_shape(rng, self.grid, p_flat=self.pr.p_flat)
            raw = np.einsum("cmn,n->cm", self.A, s["g"])
            prof = None if self.pr.noise_profile_mode == "measured" \
                else np.ones_like(NOISE_PROFILE)
            raw = apply_noise(raw, rng, self.pr.noise_lo, self.pr.noise_hi,
                              profile=prof)
            for c in range(C):  # independent residual alignment errors
                sh = int(rng.integers(-self.pr.shift_max, self.pr.shift_max + 1))
                if sh:
                    raw[c] = np.roll(raw[c], sh)
            mask = (rng.random(C) >= self.pr.drop_p).astype(np.float32)
            if mask.sum() < 8:  # keep at least a workable subset
                mask[rng.choice(C, 8, replace=False)] = 1.0
            mbar = np.maximum(raw.mean(axis=1, keepdims=True), self.pr.eps_norm)
            d = (raw / mbar * mask[:, None]).astype(np.float32)
            r_true = float(np.sqrt((s["verts"][:, :2] ** 2).sum(1)).max())
            tv, tf = s["verts"], s["faces"]
            p_t = s["p"].astype(np.float32)
            if self.pr.canonical_r:
                # curves above come from the TRUE body; the target is its canonical form
                tv = canonicalize_r(tv)
                tv, tf = hull_mesh(tv)
                gc = mesh_to_egi(tv, tf, self.grid)
                p_t = (gc / max(gc.sum(), 1e-12)).astype(np.float32)
            h = mesh_support(tv, self.grid.normals).astype(np.float32)
            r_in = r_true * float(np.exp(rng.normal(0.0, self.pr.r_jitter)))
            # rho_true comes from the TRUE mesh, not from its 1152-normal support
            # approximation, so the Dice target is not capped by the grid.
            rho = (mesh_radial(tv, tf, self.rays).astype(np.float32)
                   if self.rays is not None else np.zeros(1, dtype=np.float32))
            yield (torch.from_numpy(d), torch.from_numpy(mask),
                   torch.from_numpy(p_t), torch.from_numpy(h),
                   torch.tensor(np.log(max(r_in, 1e-6)), dtype=torch.float32),
                   torch.from_numpy(rho))


def build_model(pr: Preset, device: str) -> tuple:
    grid = make_grid(pr.n_theta, pr.n_phi)
    cameras = build_cameras()
    A, types = stack_A(grid, cameras, pr.m, c_lambert=pr.c_lambert,
                       sigma=pr.sigma, delta=pr.delta)
    op = ConvexPhotometricOperator(A, eps=pr.eps_norm)
    net = LPDNet(op, pr.n_theta, pr.n_phi, cameras + cameras, types,
                 n_iter=pr.n_iter, n_primal=pr.n_primal, n_dual=pr.n_dual,
                 ch=pr.ch, support_head=pr.support_head,
                 r_cond=pr.r_cond, gate_rank=pr.gate_rank,
                 gate_bias=pr.gate_bias).to(device)
    return net, grid, A, cameras, types


def train(pr: Preset, out_dir: str = "checkpoints", device: str | None = None,
          resume: str | None = None, dataset=None,
          warm_start_from: str | None = None) -> Path:
    device = device or auto_device()
    torch.manual_seed(pr.seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    net, grid, A, _, _ = build_model(pr, device)
    u = torch.as_tensor(grid.normals, dtype=torch.float32, device=device)
    M_rays = None
    if pr.dice_weight:
        M_rays = torch.as_tensor(
            support_ray_matrix(grid.normals, fibonacci_sphere(pr.n_rays)),
            dtype=torch.float32, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=pr.lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=pr.steps)
    step0 = 0
    if warm_start_from and not resume:
        info = warm_start(net, warm_start_from, pr.gate_rank, pr.n_primal)
        print(f"warm start from {warm_start_from}: {info['loaded']} tensors copied, "
              f"{info['grown']} widened, {len(info['skipped'])} new "
              f"(gate weights): {info['skipped'][:4]}", flush=True)
    if resume:
        ck = torch.load(resume, map_location=device)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step0 = ck["step"]
    # bf16, not fp16. The forward divides by mean.clamp_min(1e-3), so a curve the gate has
    # nearly closed produces ratios of order 1e3-1e5; fp16 tops out at 65504 and returns
    # inf, which is where the NaNs came from. bf16 carries fp32's exponent range (~3e38)
    # at fp16's speed on tensor cores, so that overflow cannot happen, and it needs no
    # GradScaler -- removing the scale/unscale sawtooth that made the failures look
    # intermittent. fp16 stays available for hardware without bf16.
    use_amp = pr.amp and device == "cuda"
    amp_dtype = torch.float16
    if use_amp and pr.amp_dtype == "bf16":
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            print("bf16 unsupported on this GPU; falling back to fp16", flush=True)
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype is torch.float16)
    if use_amp:
        print(f"autocast dtype: {amp_dtype}", flush=True)
    dl = DataLoader(dataset if dataset is not None else SyntheticCurves(A, grid, pr),
                    batch_size=pr.batch,
                    num_workers=pr.num_workers, pin_memory=(device == "cuda"))
    it, t0 = iter(dl), time.time()
    N = grid.n
    nonfinite, inf_grads, MAX_NONFINITE = 0, 0, 10
    for step in range(step0, pr.steps):
        batch = [x.to(device, non_blocking=True) for x in next(it)]
        d, mask, p = batch[0], batch[1], batch[2]
        h_true = batch[3] if len(batch) > 3 else None
        log_r = batch[4] if len(batch) > 4 else None
        rho_true = batch[5] if len(batch) > 5 else None
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp,
                            dtype=amp_dtype):
            pred = net(d, mask, log_r) if pr.r_cond else net(d, mask)
            p_hat = pred[0]
            mse = N * ((p_hat - p) ** 2).sum(dim=1).mean()
            closure = (p_hat @ u).pow(2).sum(dim=1).mean()
            loss = pr.egi_weight * (mse + pr.closure_weight * closure)
            h_mse = None
            if pr.support_head and h_true is not None:
                # sum over directions (h ~ O(1) per direction), mean over batch --
                # comparable in magnitude to the N*MSE used for the EGI.
                h_mse = ((pred[2] - h_true) ** 2).sum(dim=1).mean()
                loss = loss + pr.h_mse_weight * h_mse
            dice_l = None
            if pr.dice_weight and rho_true is not None and rho_true.shape[1] > 1:
                # the scoring function itself. Computed in fp32 even under autocast:
                # rho is a ratio of a max, and fp16 rounding there is visible in the
                # third decimal of the score.
                with torch.autocast(device_type=d.device.type, enabled=False):
                    dice_l = torch_dice_loss(pred[2].float(), rho_true.float(),
                                             M_rays, chunk=pr.dice_chunk)
                loss = loss + pr.dice_weight * dice_l
        # A non-finite loss must never reach the weights. clip_grad_norm_ cannot help --
        # it rescales by a norm that is itself NaN -- and once the weights are NaN every
        # later step and every later checkpoint is poisoned. So check, drop the step, and
        # abort if it is not a one-off: a run that skips thousands of steps is
        # not training, it is pretending to.
        if not torch.isfinite(loss):
            # A non-finite FORWARD is a genuine fault -- the model produced a number that
            # does not exist -- and no scaler can repair it. Drop the step so the weights
            # stay clean, and abort if it recurs: a run that skips steps is not training,
            # it is pretending to. Nothing has been scaled or unscaled yet at this point,
            # so the GradScaler's state is untouched and `continue` is safe.
            nonfinite += 1
            opt.zero_grad(set_to_none=True)
            print(f"step {step:>7d}  NON-FINITE loss, step dropped "
                  f"(#{nonfinite}/{MAX_NONFINITE})", flush=True)
            if nonfinite > MAX_NONFINITE:
                raise RuntimeError(
                    f"{nonfinite} non-finite losses by step {step}; aborting rather than "
                    "training through them")
            sched.step()
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        # Under fp16 an occasional inf gradient is the GradScaler's designed operating
        # point, not a fault: step() inspects found_inf and skips the update itself. So
        # count it for visibility and let the scaler do its job -- never `continue` here,
        # which would strand the scaler between unscale_ and update.
        if not torch.isfinite(gnorm):
            inf_grads += 1
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % pr.log_every == 0:
            with torch.no_grad():
                cos = torch.nn.functional.cosine_similarity(p_hat, p, dim=1).mean()
            hs = f"  hmse {h_mse.item():.4e}" if h_mse is not None else ""
            hs += f"  dice {1.0-dice_l.item():.4f}" if dice_l is not None else ""
            print(f"step {step:>7d}{hs}  loss {loss.item():.4e}  mse {mse.item():.4e}  "
                  f"closure {closure.item():.3e}  cos {cos.item():.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s"
                  + (f"  infgrad {inf_grads}" if inf_grads else ""), flush=True)
        if step and step % pr.ckpt_every == 0 or step == pr.steps - 1:
            _assert_finite(net, step)
            path = out / f"lpd_{pr.name}_step{step}.pt"
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step,
                        "preset": asdict(pr)}, path)
    final = out / f"lpd_{pr.name}_final.pt"
    _assert_finite(net, pr.steps)
    torch.save({"model": net.state_dict(), "preset": asdict(pr), "step": pr.steps}, final)
    return final


def _assert_finite(net, step: int) -> None:
    """Refuse to write a poisoned checkpoint. A NaN weight on disk is worse than a
    crashed run: it looks like a result and silently ruins everything downstream."""
    bad = [k for k, v in net.state_dict().items()
           if v.is_floating_point() and not torch.isfinite(v).all()]
    if bad:
        raise RuntimeError(f"step {step}: non-finite weights in {len(bad)} tensors "
                           f"({bad[:4]}); refusing to checkpoint")


# Buffers that build_model() regenerates exactly from the preset, so they never need to
# be carried in a checkpoint. op.A alone is (56, 360, 1152) float32 = 92.9 MB, against
# 2.4 MB of actual learned weights -- 97% of a saved checkpoint is a tensor that is a
# pure function of the preset.
REGENERABLE_BUFFERS = ("op.A", "tags", "coords")


def warm_start(net, ckpt_path: str, gate_rank: int, n_primal: int) -> dict:
    """Load ungated weights into a gated network, exactly.

    The primal block's input is cat([f, back, coords]) and `back` widens from 1 channel
    to `gate_rank`, so the first conv's weight grows in its input dimension and the
    coords channels shift. A plain load_state_dict therefore fails, and a naive
    leading-slice copy would silently feed coords into the wrong filters.

    The mapping is explicit: f and back-rank-0 keep their slots, the new back ranks get
    ZERO weight, and coords move to their new offset. Zero weight on the new ranks means
    the network computes exactly what the ungated one did -- combined with gate_scale=0
    (rank 0 gated to 1, higher ranks to 0), the warm-started model is bit-identical to
    the checkpoint it came from, at any rank.
    """
    src = torch.load(ckpt_path, map_location="cpu")["model"]
    tgt = net.state_dict()
    loaded, grown, skipped = 0, 0, []
    for k, v in src.items():
        if k not in tgt:
            skipped.append(k)
            continue
        w = tgt[k]
        if w.shape == v.shape:
            w.copy_(v)
            loaded += 1
        elif (k.startswith("primals.") and k.endswith("c1.weight")
              and w.shape[1] == v.shape[1] + gate_rank - 1):
            n_c = v.shape[1] - n_primal - 1          # coords (+ optional log R) channels
            w.zero_()
            w[:, :n_primal] = v[:, :n_primal]                     # f
            w[:, n_primal] = v[:, n_primal]                       # back, rank 0
            w[:, n_primal + gate_rank:] = v[:, n_primal + 1:]     # coords, shifted
            grown += 1
        else:
            skipped.append(k)
    net.load_state_dict(tgt)
    return {"loaded": loaded, "grown": grown, "skipped": skipped}


def load_net(ckpt_path: str, device: str | None = None) -> tuple:
    device = device or auto_device()
    ck = torch.load(ckpt_path, map_location=device)
    # Ignore preset keys this version no longer defines, so a checkpoint keeps loading
    # if a field is ever retired. Without this, deleting a single Preset field silently
    # breaks every checkpoint ever written.
    known = {f.name for f in fields(Preset)}
    stale = sorted(set(ck["preset"]) - known)
    if stale:
        print(f"note: ignoring retired preset keys {stale}")
    pr = Preset(**{k: v for k, v in ck["preset"].items() if k in known})
    net, grid, A, cameras, types = build_model(pr, device)
    missing, unexpected = net.load_state_dict(ck["model"], strict=False)
    # strict=False is only safe because the missing keys are checked: anything
    # other than a regenerable buffer means the checkpoint really is incomplete.
    bad = [k for k in missing if k not in REGENERABLE_BUFFERS]
    if bad or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing {bad}, unexpected {list(unexpected)}")
    net.eval()
    return net, pr, grid
