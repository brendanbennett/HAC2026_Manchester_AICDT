#!/usr/bin/env python3
"""Export everyday 3-D printable objects from Thingi10K for the library's "object" family.

    pip install thingi10k
    python scripts/fetch_objects.py --n 600         # into dataset/shape_models/objects

A secret model need not be an asteroid; a figure or a household object is as printable.
Thingi10K is ten thousand models people have printed, with the geometry checked per file.
The ones taken here are closed, solid, single-piece and manifold, with a bounded vertex
count; a body with up to --genus holes is allowed, since a printed object can have a handle.
Only models under a licence that allows reuse are taken (public domain, CC0, CC BY), and
models tagged as text, logos, signs, gears, screws, brackets and the like are left out, since
flat lettering and machine parts are unlikely to be put on the turntable. One model per
Thingiverse "thing", so that a family of variants does not crowd the set. The chosen models
are written as OBJ files into a subdirectory of the shape-model directory, where
scripts/build_shape_library.py picks them up as the "object" family. The first run downloads
the dataset (several gigabytes) into thingi10k's cache.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LICENCE_OK = ("public domain", "cc0", "attribution", "cc by")
LICENCE_BAD_WORDS = {"nc", "nd", "sa", "noncommercial", "noderivatives", "sharealike"}
LICENCE_BAD_PHRASES = ("non commercial", "no derivatives", "share alike")
TAGS_BAD = ("text", "logo", "sign", "nameplate", "keychain", "gear", "bearing", "bracket",
            "screw", "bolt", "phone case", "coin", "badge")


def _allowed(entry) -> bool:
    """A reusable licence and nothing that says lettering or machine part."""
    lic = str(entry.get("license") or "").lower().replace("-", " ").replace("_", " ")
    if not any(k in lic for k in LICENCE_OK):
        return False
    if set(lic.split()) & LICENCE_BAD_WORDS or any(p in lic for p in LICENCE_BAD_PHRASES):
        return False
    text = " ".join([str(entry.get("name") or "")] + [str(t) for t in (entry.get("tags") or [])])
    return not any(k in text.lower() for k in TAGS_BAD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset/shape_models/objects")
    ap.add_argument("--n", type=int, default=600, help="how many objects to export")
    ap.add_argument("--seed", type=int, default=0, help="which of the eligible ones")
    ap.add_argument("--min-vertices", type=int, default=300)
    ap.add_argument("--max-vertices", type=int, default=60000)
    ap.add_argument("--genus", type=int, default=2, help="largest number of holes allowed")
    ap.add_argument("--cache-dir", default=None, help="thingi10k's download cache")
    a = ap.parse_args()
    try:
        import thingi10k
    except ImportError:
        raise SystemExit("pip install thingi10k first")

    thingi10k.init(cache_dir=a.cache_dir)
    ds = thingi10k.dataset(closed=True, solid=True, manifold=True, num_components=1,
                           num_vertices=(a.min_vertices, a.max_vertices), genus=(0, a.genus))
    print(f"  {len(ds)} objects in Thingi10K pass the geometry filters", flush=True)
    order = np.random.default_rng(a.seed).permutation(len(ds))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    seen_things = set()
    for k in order:
        if n_ok >= a.n:
            break
        e = ds[int(k)]
        if not _allowed(e) or e["thing_id"] in seen_things:
            continue
        seen_things.add(e["thing_id"])
        dst = out / f"thingi_{int(e['file_id']):06d}.obj"
        if dst.exists():
            n_ok += 1
            continue
        try:
            v, f = thingi10k.load_file(e["file_path"])
        except Exception as exc:                              # noqa: BLE001
            print(f"  skipped {e['file_id']}: {exc}", flush=True)
            continue
        v = np.asarray(v, float); f = np.asarray(f, np.int64)
        if f.shape[1] != 3:
            continue
        with open(dst, "w") as fh:
            fh.write(f"# Thingi10K file {e['file_id']}, thing {e['thing_id']}, "
                     f"{e.get('name', '')}, licence: {e.get('license', '')}\n")
            fh.writelines(f"v {x:.6g} {y:.6g} {z:.6g}\n" for x, y, z in v)
            fh.writelines(f"f {i + 1} {j + 1} {k + 1}\n" for i, j, k in f)
        n_ok += 1
    print(f"  {n_ok} objects in {out}", flush=True)


if __name__ == "__main__":
    main()
