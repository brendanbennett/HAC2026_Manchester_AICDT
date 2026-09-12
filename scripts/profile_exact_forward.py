#!/usr/bin/env python3
"""Time ExactForward on one GA candidate mesh, to size a --forward-model exact tuning budget
before committing to one.

Measures two things separately, since they cost very differently and only one of them is
GPU-accelerated:

  - _prepare(): the interreflection setup (mesh decimation to radiosity patches, patch
    visibility via ray-casting, form factors, KD-tree face-to-patch map). Confirmed on a
    laptop CPU (see notes/ or ask -- this was measured directly) to be cheap, under a second,
    and CPU-bound regardless of device: decimate() and _visibility_matrix() are plain
    trimesh/numpy, with no device argument at all (hac26/forward/mesh/exact.py:94-102,
    hac26/forward/mesh/radiosity.py:44-73).
  - raw_curves(): the actual rendering, geometries x phases frames at the configured
    resolution. This is what nvdiffrast accelerates; on the CPU/software backend
    (hac26/forward/shared/software_raster.py) it is a brute-force O(pixels x triangles) scan
    with no acceleration structure, and dominates total per-candidate cost by roughly two
    orders of magnitude over _prepare() even at a small fraction of production scale.

Needs only results/lpd/ (git-tracked, small) as a mesh source by default, not results/genetic/
(disposable local experiment output) -- see default_mesh() below.

Run small first (defaults) to sanity-check timing before requesting the full production
geometry/phase count, which was measured to take multiple minutes per call on a laptop CPU.

    python scripts/profile_exact_forward.py --device cuda
    python scripts/profile_exact_forward.py --device cuda --n-geoms 28 --m 50   # production scale
    python scripts/profile_exact_forward.py --device cuda --backend software   # force no nvdiffrast
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import trimesh

from hac26.conventions import psi_grid
from hac26.forward.mesh.exact import ExactForward, RenderConfig
from hac26.forward.mesh.instrument import Instrument
from hac26.stl_io import load_stl

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION = REPO_ROOT / "models/instrument_calibration.pt"


def default_mesh(model: int, target_faces: int) -> tuple:
    """A representative GA candidate mesh, decimated from results/lpd/AsteroidNN.stl (the
    only mesh source this script needs that is both git-tracked and small -- unlike
    results/genetic/, which is disposable local experiment output not worth transferring to
    a fresh machine). Same decimate-after-merge approach as
    scripts/run_final_ga_all_models.py's decimate_lpd_mesh, just without writing the result
    to disk since this script only needs it in memory."""
    path = REPO_ROOT / f"results/lpd/Asteroid{model:02d}.stl"
    if not path.exists():
        raise SystemExit(f"{path} not found -- pass --mesh explicitly, or run the LPD "
                         f"reconstruction for model {model} first")
    verts, faces = load_stl(str(path))
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if len(mesh.faces) > target_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def load_instrument(path: Path, device: str) -> Instrument:
    if path.exists():
        try:
            return Instrument.load(str(path), device=device)
        except RuntimeError as exc:
            print(f"warning: could not load {path} ({exc}); using an uncalibrated Instrument "
                 f"instead -- fine for timing, not for real Dice/fitness numbers", flush=True)
    else:
        print(f"warning: {path} not found; using an uncalibrated Instrument -- fine for "
             f"timing, not for real Dice/fitness numbers", flush=True)
    return Instrument().to(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", default=None,
                    help="a representative GA candidate mesh (decimated start mesh, not the "
                         "full-resolution LPD output) -- this is what the GA actually renders. "
                         "Default: decimate results/lpd/Asteroid<model>.stl on the fly (see "
                         "--model, --target-faces)")
    ap.add_argument("--model", type=int, default=1,
                    help="which results/lpd/AsteroidNN.stl to decimate when --mesh is not given")
    ap.add_argument("--target-faces", type=int, default=2000,
                    help="matches scripts/run_final_ga_all_models.py's default")
    ap.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    ap.add_argument("--device", default="cuda", help="cpu/cuda/mps")
    ap.add_argument("--backend", default=None,
                    help="nvdiffrast (default) or software; software is the CPU stand-in and "
                         "is expected to be ~2 orders of magnitude slower")
    ap.add_argument("--n-geoms", type=int, default=4,
                    help="cameras to render; production (scripts/reconstruct_genetic.py "
                         "default) uses all of them -- 28")
    ap.add_argument("--m", type=int, default=8,
                    help="phase samples; production default is 50")
    ap.add_argument("--radiosity-faces", type=int, default=200,
                    help="matches scripts/reconstruct_genetic.py's --exact-radiosity-faces "
                         "default for the GA, lower than RenderConfig's own default of 600")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--production-geoms", type=int, default=28,
                    help="used only to print an extrapolated production-scale estimate")
    ap.add_argument("--production-m", type=int, default=50)
    a = ap.parse_args()

    if a.mesh is not None:
        verts, faces = load_stl(a.mesh)
        mesh_desc = a.mesh
    else:
        verts, faces = default_mesh(a.model, a.target_faces)
        mesh_desc = f"results/lpd/Asteroid{a.model:02d}.stl, decimated to {a.target_faces} faces"
    print(f"mesh: {mesh_desc} -- {len(verts)} verts, {len(faces)} faces", flush=True)

    inst = load_instrument(Path(a.calibration), a.device)
    cfg = RenderConfig(radiosity_faces=a.radiosity_faces)
    psi = psi_grid(frames=a.m)
    fwd = ExactForward(inst, psi, config=cfg, device=a.device, backend=a.backend)
    print(f"device={a.device}, backend={a.backend or 'nvdiffrast (default)'}, "
         f"radiosity_faces={a.radiosity_faces}", flush=True)

    v = torch.as_tensor(np.ascontiguousarray(verts, dtype=np.float32), device=a.device)
    f = torch.as_tensor(np.ascontiguousarray(faces, dtype=np.int64), device=a.device)
    geoms = list(range(a.n_geoms))

    print(f"\n--- _prepare() only: radiosity setup, CPU-bound regardless of --device ---",
         flush=True)
    prepare_times = []
    for i in range(a.repeats):
        t0 = time.perf_counter()
        fwd._prepare(v, f)
        t1 = time.perf_counter()
        prepare_times.append(t1 - t0)
        print(f"  call {i}: {t1 - t0:.3f}s", flush=True)

    print(f"\n--- raw_curves(): n_geoms={a.n_geoms}, m={a.m} "
         f"({a.n_geoms * a.m} geometry-phase pairs) ---", flush=True)
    curve_times = []
    for i in range(a.repeats):
        t0 = time.perf_counter()
        with torch.no_grad():
            raw = fwd.raw_curves(v, f, geoms=geoms)
        t1 = time.perf_counter()
        curve_times.append(t1 - t0)
        print(f"  call {i}: {t1 - t0:.3f}s, out shape {tuple(raw.shape)}", flush=True)

    # First call often includes one-time setup (context/kernel compilation, allocator
    # warmup); the steady-state mean is the one worth extrapolating from when repeats > 1.
    steady = curve_times[1:] if len(curve_times) > 1 else curve_times
    mean_call = float(np.mean(steady))
    pairs_tested = a.n_geoms * a.m
    pairs_production = a.production_geoms * a.production_m
    scale = pairs_production / pairs_tested

    # raw_curves() = _prepare() [CPU-bound, independent of how many frames are then rendered]
    # + rendering pairs_tested frames. Only the second part scales with pair count -- scaling
    # the whole call (as an earlier version of this script did) treats the fixed setup cost
    # as if it multiplied by `scale` too, which overstates production cost whenever setup is
    # a non-trivial share of a small test call (it usually is: this mesh's _prepare() was
    # ~0.7-0.8s on a laptop CPU, comparable to or larger than a tiny few-pair render).
    prepare_mean = float(np.mean(prepare_times))
    render_only = max(mean_call - prepare_mean, 0.0)
    est_production = prepare_mean + render_only * scale
    est_production_naive = mean_call * scale   # what scaling the whole call would have said

    print(f"\n--- extrapolation ---")
    print(f"steady-state mean per call: {mean_call:.3f}s for {pairs_tested} geometry-phase "
         f"pairs ({prepare_mean:.3f}s _prepare + {render_only:.3f}s render)")
    print(f"scaling only the render part linearly by pair count to production "
         f"({a.production_geoms} geoms x {a.production_m} phases = {pairs_production} pairs, "
         f"{scale:.1f}x more), _prepare held fixed:")
    print(f"  estimated per-candidate cost: {est_production:.1f}s "
         f"({est_production / 60:.1f} min) -- naively scaling the whole call instead would "
         f"say {est_production_naive:.1f}s ({est_production_naive / est_production:.1f}x "
         f"higher, wrong whenever _prepare is a non-trivial share of the tested call)")
    for pop, gens, label in [(33, 134, "tuned convex population/generations"),
                             (10, 20, "a much smaller exact-model trial")]:
        total_s = est_production * pop * gens
        print(f"  one trial at population={pop}, generations={gens} ({label}): "
             f"{total_s / 3600:.1f}h for {pop * gens} evaluations")
    budget_s = 8 * 3600
    print(f"  candidates fitting in an 8h budget at this rate: "
         f"~{int(budget_s / est_production)}")


if __name__ == "__main__":
    main()
