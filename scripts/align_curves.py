#!/usr/bin/env python3
"""Realign each azimuth's measured curves to the Blender reference, and write a new data dir.

    python scripts/align_curves.py --out dataset/aligned
    python scripts/align_curves.py --out dataset/aligned --models 1 2 3 --report

Each azimuth of the real measurement is a separate video, trimmed independently
(`CAM1_2A_0_trim.mp4`), so each carries its own start phase. The organisers realigned model
1's curves per azimuth on 17/25 Aug 2026 for exactly this reason -- four columns at a time,
by a whole-frame offset that differed per azimuth -- and a residual offset is still present in
every model.

The calibration fits **one** start phase per body, so a per-azimuth offset cannot be
represented and lands in the residual as forward-model error. It is absorbed into eta, which
inflates the noise floor until a convex body explains everything: on model 3 the convex answer
starts at chi 1.24 and one gradient step takes it under the 1.0 stop threshold, leaving no
misfit to drive any concavity. DAMIT's renderer does not have this problem because it lags
each of the 28 curves independently, which is why it reproduces the Blender curves to 0.005
where ours needs eta = 0.086.

The offset is measured against the **Blender** curves, which ship with all ten models and are
rendered from the true shape at a single consistent phase convention. So no shape model is
needed and this works on the secret models too.

A phase offset is a rigid shift of a curve. It cannot imitate a concavity, which changes a
curve's shape rather than its position, so fitting one does not buy misfit with free
parameters -- the objection `calibrate.phase_offset_report` raises against fitting seven more
parameters per body. What it can do is fail on a body whose curves are nearly flat, where the
shift is unidentifiable: model 8 is nearly spherical and its measured offsets scatter over 72
degrees against 2.5-16.5 for every other model. `--max-shift` refuses those rather than
applying a fitted number that is really noise.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.data_io import load_model_curves                              # noqa: E402

N_AZ, N_CH = 7, 4
AZIMUTHS = (0, 45, 90, 135, 225, 270, 315)


def _mean_norm(c):
    m = c.mean(axis=0, keepdims=True)
    return c / np.where(np.abs(m) > 0, m, 1.0)


def _resample(c, P):
    """[T, ...] -> [P, ...] by periodic-linear interpolation."""
    T = c.shape[0]
    src = np.arange(T + 1)
    wrapped = np.concatenate([c, c[:1]], axis=0).reshape(T + 1, -1)
    at = np.arange(P) * (T / P)
    out = np.stack([np.interp(at, src, wrapped[:, j]) for j in range(wrapped.shape[1])],
                   axis=1)
    return out.reshape(P, *c.shape[1:])


def read_raw(path: Path) -> np.ndarray:
    raw = np.loadtxt(path, delimiter=",")
    if raw.shape[1] != 29:
        raise SystemExit(f"{path}: expected 29 columns, got {raw.shape[1]}")
    return raw


def offsets_for(real: np.ndarray, blend: np.ndarray, P: int = 720) -> np.ndarray:
    """Degrees each azimuth of `real` must rotate by to sit on `blend`."""
    r = _mean_norm(_resample(real[:, 1:].reshape(len(real), N_AZ, N_CH), P))
    b = _mean_norm(_resample(blend[:, 1:].reshape(len(blend), N_AZ, N_CH), P))
    out = np.zeros(N_AZ)
    for az in range(N_AZ):
        best = (np.inf, 0)
        for lag in range(-P // 8, P // 8 + 1):
            d = float(np.sqrt(np.mean((r[:, az] - np.roll(b[:, az], lag, axis=0)) ** 2)))
            if d < best[0]:
                best = (d, lag)
        out[az] = best[1] * 360.0 / P
    return out


def apply_offsets(raw: np.ndarray, deg: np.ndarray) -> np.ndarray:
    """Roll each azimuth's four columns by its offset, at the file's own frame rate."""
    T = len(raw)
    cur = raw[:, 1:].reshape(T, N_AZ, N_CH).copy()
    idx = np.arange(T)
    for az in range(N_AZ):
        # np.roll(x, k)[i] = x[i-k], so `offsets_for` returning lag means real[i] ~ blend[i-lag]
        # and the aligned curve is real[i+lag]: the sample point moves *forward* by the offset.
        # The opposite sign adds the misalignment instead of removing it -- it took eta from
        # 0.1237 to 0.1430 and doubled the residual spread before this was caught.
        shift = deg[az] * T / 360.0           # move the real curve onto the reference
        at = (idx + shift) % T
        i0 = np.floor(at).astype(int) % T
        w = (at - np.floor(at))[:, None]
        cur[:, az] = cur[i0, az] * (1 - w) + cur[(i0 + 1) % T, az] * w
    out = raw.copy()
    out[:, 1:] = cur.reshape(T, N_AZ * N_CH)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset/raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", type=int, default=list(range(1, 11)))
    ap.add_argument("--max-shift", type=float, default=20.0,
                    help="refuse a body whose azimuths disagree by more than this many "
                         "degrees; its curves are too flat to locate a phase")
    ap.add_argument("--report", action="store_true", help="measure but write nothing")
    a = ap.parse_args()

    src_root, out_root = Path(a.data_dir), Path(a.out)
    print(f"{'model':>5} " + " ".join(f"{d:>7}" for d in AZIMUTHS) + f" {'spread':>8}  action")
    for M in a.models:
        d = load_model_curves(a.data_dir, M, m=360)
        srcs = {k: Path(v) for k, v in d["files"].items()}
        if not srcs:
            print(f"{M:>5}  no curves found, skipped"); continue
        real_i = read_raw(srcs["intensity"])
        blend_p = srcs["intensity"].with_name(
            srcs["intensity"].name.replace(".txt", "_blender.txt"))
        if not blend_p.exists():
            print(f"{M:>5}  no blender reference, skipped"); continue
        off = offsets_for(real_i, read_raw(blend_p))
        spread = float(off.max() - off.min())
        ok = spread <= a.max_shift
        print(f"{M:>5} " + " ".join(f"{o:>+7.1f}" for o in off) +
              f" {spread:>8.1f}  {'align' if ok else 'REFUSED (curves too flat)'}")
        if a.report:
            continue
        mdir = srcs["intensity"].parent
        rel = mdir.relative_to(src_root)
        dst = out_root / rel
        dst.mkdir(parents=True, exist_ok=True)
        for kind, p in srcs.items():
            raw = read_raw(p)
            new = apply_offsets(raw, off) if ok else raw
            np.savetxt(dst / p.name, new, delimiter=",", fmt="%.6f")
            bp = p.with_name(p.name.replace(".txt", "_blender.txt"))
            if bp.exists():
                shutil.copy2(bp, dst / bp.name)
        # the shape file sits one level up and the loaders look for it there
        for extra in (src_root / rel.parts[0]).glob("*.stl"):
            link = out_root / rel.parts[0] / extra.name
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                link.symlink_to(extra.resolve())
    if not a.report:
        print(f"\nwrote {out_root}")


if __name__ == "__main__":
    main()
