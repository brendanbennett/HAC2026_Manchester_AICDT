#!/usr/bin/env python3
"""Fidelity check of TEAM-GENERATED curves against the published lab curves.

Run this FIRST when the team dataset lands. For each public model 1-3 it compares
a provided generated-curves file against the official real and Blender curves,
reporting per-type MSE and the best global cyclic shift (nonzero shift => phase
convention mismatch; mirrored ordering => wrong sigma/delta).

The --gen npz may be either the legacy single-array layout ('curves' (56, m)) or
the team make_dataset schema ('intensity'/'binary' (frames, 28) + 'azimuth'/
'elevation' (28,)); the latter is reordered to challenge-camera order and stacked
into (56, m) exactly as adapter.load_pairs does before training.

    python fidelity_check.py --gen path/to/Asteroid03_curves.npz --model 3 \
        --data-dir ../data/raw
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hac26.data_io import load_model_curves  # noqa: E402
from forward_models.convex_egi import normalize_np  # noqa: E402


def load_gen_curves(path: str) -> np.ndarray:
    """Return generated curves as (56, m). Accepts the legacy 'curves' array or
    the team make_dataset schema (mirrors adapter.load_pairs' conversion)."""
    z = np.load(path, allow_pickle=True)
    if "curves" in z:
        return z["curves"].astype(float)
    from hac26.geometry import build_cameras  # noqa: E402
    cols = list(zip(np.round(z["azimuth"], 3), np.round(z["elevation"], 3)))
    order, used = [], set()
    for cam in build_cameras():
        j = next(k for k, (a, e) in enumerate(cols) if k not in used
                 and a == round(cam.azimuth_deg, 3)
                 and abs(e - cam.elevation_deg) < 0.51)
        order.append(j)
        used.add(j)
    return np.concatenate([z["intensity"].T[order], z["binary"].T[order]]).astype(float)


def best_shift_mse(a: np.ndarray, b: np.ndarray) -> tuple:
    m = a.shape[-1]
    corr = np.fft.irfft(np.fft.rfft(a, axis=1).conj() * np.fft.rfft(b, axis=1),
                        n=m, axis=1).sum(0)
    s = int(np.argmin((a ** 2).sum() + (b ** 2).sum() - 2 * corr))
    return s if s <= m // 2 else s - m, float(((np.roll(a, -s, axis=1) - b) ** 2).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--model", type=int, required=True)
    ap.add_argument("--data-dir", default="../data/raw")
    args = ap.parse_args()
    gen = normalize_np(load_gen_curves(args.gen))
    m = gen.shape[-1]
    for label, blender in (("real", False), ("blender", True)):
        ref = load_model_curves(args.data_dir, args.model, m=m, use_blender=blender)
        if not ref["files"]:
            continue
        for name, sl in (("intensity", slice(0, 28)), ("binary", slice(28, 56))):
            s, mse = best_shift_mse(gen[sl], ref["curves"][sl])
            print(f"model {args.model} vs {label:7s} {name:9s}: "
                  f"best global shift {s:+4d} frames, MSE {mse:.5f}")
    print("interpret: |shift|>3 => phase mismatch; MSE >> real-vs-blender MSE of the "
          "same model => geometry/scattering mismatch (check sigma=-1, delta=+1, "
          "camera table, thresholds).")


if __name__ == "__main__":
    main()
