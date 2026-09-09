#!/usr/bin/env python3
"""Check the downloaded challenge data against dataset/MANIFEST.sha256.

The README tells you to do this and nothing did it, which is how the repository came to pin a
mixed snapshot: models 2 and 3 were refreshed by hand after the organisers re-rendered the
Blender curves on 29 July 2026 (the `.stale29jul` backups that used to be in the manifest are
the trace of it) and model 1 was missed, so its four curve files stayed on the pre-update
versions the organisers have since realigned. A single start phase cannot absorb that -- the
realignment is a different whole-frame shift per azimuth -- so it lands in the calibration's
residuals looking like forward-model error. `calibrate.py` now reports a per-azimuth phase
offset, which is the other half of catching this.

The manifest records paths under `data/raw/`; the README puts the data in `dataset/raw/`.
`--data-dir` is the root the manifest's paths are resolved against, so either layout works.

    python scripts/check_data.py                       # dataset/raw
    python scripts/check_data.py --data-dir data/raw
    python scripts/check_data.py --write               # regenerate the manifest from what is
                                                       # on disk, after a deliberate refresh

Exits non-zero if anything is missing or has changed.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

MANIFEST = Path("dataset/MANIFEST.sha256")
PREFIX = "data/raw/"          # the path prefix the manifest was written with


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def entries(manifest: Path):
    """(relative path, hash) for every line, with the manifest's own prefix stripped."""
    out = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(None, 1)
        rel = rel.strip()
        out.append((rel[len(PREFIX):] if rel.startswith(PREFIX) else rel, digest))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset/raw",
                    help="root the manifest's paths are resolved against")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--write", action="store_true",
                    help="rewrite the manifest from what is on disk instead of checking it. "
                         "Only after a refresh you meant to do: it makes whatever you have "
                         "the reference, including a stale copy")
    a = ap.parse_args()
    root = Path(a.data_dir)
    if not root.is_dir():
        print(f"{root} is not a directory; put the challenge data there first")
        return 2

    if a.write:
        files = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))
        a.manifest.parent.mkdir(parents=True, exist_ok=True)
        a.manifest.write_text("".join(
            f"{sha256(p)}  {PREFIX}{p.relative_to(root)}\n" for p in files))
        print(f"wrote {a.manifest} from {len(files)} files under {root}")
        return 0

    missing, changed, ok = [], [], 0
    for rel, digest in entries(a.manifest):
        path = root / rel
        if not path.exists():
            missing.append(rel)
        elif sha256(path) != digest:
            changed.append(rel)
        else:
            ok += 1

    listed = {rel for rel, _ in entries(a.manifest)}
    extra = sorted(str(p.relative_to(root)) for p in root.rglob("*")
                   if p.is_file() and not p.name.startswith(".")
                   and str(p.relative_to(root)) not in listed)

    print(f"{ok} of {ok + len(missing) + len(changed)} files match {a.manifest}")
    for label, items in (("missing", missing), ("changed", changed), ("not in the manifest", extra)):
        if items:
            print(f"\n{len(items)} {label}:")
            for s in items[:20]:
                print(f"   {s}")
            if len(items) > 20:
                print(f"   ... and {len(items) - 20} more")
    if changed:
        print("\nA changed file is either a download the organisers have since replaced or a\n"
              "manifest entry that was never refreshed. Check the challenge page's News &\n"
              "Updates before assuming the manifest is right, then rerun with --write.")
    if missing or changed:
        print("\nThe calibration and everything downstream of it read these files.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
