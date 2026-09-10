#!/usr/bin/env python3
"""Decide, per scored model, whether a refined body replaces the convex answer.

    python scripts/select_answers.py --refined results/map --ratio 0.7

A refinement is accepted for a model when its misfit on the geometries held out of its fit,
measured on the body as written (`chi_held_export` in the refinement's JSON), is at most
`--ratio` times the convex answer's misfit on the same geometries (`chi_held_convex`), and
the file passes the submission check. The ratio is not a default of this script because it
is a measured quantity: it comes from the public model whose refinement improved the score
against the released shape, and a refinement of a scored model is trusted only where it
beats its convex answer by at least as much as that one did. Without a public model on
which the refinement passed, there is no ratio and the convex answers stand.

The accepted files are copied over the convex answers under results/submission, which
scripts/make_submission.py regenerates in seconds, and results/submission/selection.json
records the choice and the numbers behind it, so the submitted directory says what it
contains.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_submission import inspect                       # noqa: E402
from hac26.conventions import CYLINDER_R, PUBLIC_MODELS    # noqa: E402
from reconstruct import answer_path                        # noqa: E402

SELECTION_FILE = "results/submission/selection.json"


def decide(meta: dict, ratio: float) -> tuple:
    """(accept, reason) for one refinement's JSON. A refinement fitted without held-out
    geometries has no honest number and is refused."""
    held = meta.get("chi_held_export")
    base = meta.get("chi_held_convex")
    if not meta.get("held_out"):
        return False, "no geometries were held out of the fit"
    if held is None or base is None or not (held < float("inf")):
        return False, "the written body has no held-out misfit"
    if held > ratio * base:
        return False, f"held-out misfit {held:.3f} above {ratio:g} x convex {base:.3f}"
    return True, f"held-out misfit {held:.3f} at or below {ratio:g} x convex {base:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refined", required=True,
                    help="directory of Asteroid<NN>.stl and .json written by reconstruct_map.py")
    ap.add_argument("--ratio", type=float, required=True,
                    help="largest accepted held-out misfit of a refinement, as a fraction of "
                         "the convex answer's; measured on the public model that passed")
    ap.add_argument("--models", nargs="+", type=int,
                    default=[M for M in range(1, 11) if M not in PUBLIC_MODELS])
    a = ap.parse_args()
    if not 0.0 < a.ratio <= 1.0:
        raise SystemExit("--ratio must lie in (0, 1]: a refinement that fits the held-out "
                         "geometries no better than the convex answer is not accepted")

    selection = {"ratio": a.ratio, "refined_dir": a.refined, "models": {}}
    for M in a.models:
        target = answer_path(M)
        stl = Path(a.refined) / f"Asteroid{M:02d}.stl"
        js = stl.with_suffix(".json")
        entry = {"answer": "convex"}
        if not (stl.exists() and js.exists()):
            entry["reason"] = "no refinement written"
        else:
            meta = json.loads(js.read_text())
            accept, reason = decide(meta, a.ratio)
            check = inspect(stl, CYLINDER_R[M])
            if accept and check["fails"]:
                accept, reason = False, "refined file fails the submission check: " + \
                    "; ".join(check["fails"])
            entry.update({"reason": reason, "chi_held_convex": meta.get("chi_held_convex"),
                          "chi_held_export": meta.get("chi_held_export"),
                          "held_out": meta.get("held_out")})
            if accept:
                shutil.copyfile(stl, target)
                entry["answer"] = "refined"
        selection["models"][M] = entry
        print(f"model {M:>2}: {entry['answer']:8s} {entry['reason']}", flush=True)
    Path(SELECTION_FILE).write_text(json.dumps(selection, indent=2))
    print(f"wrote {SELECTION_FILE}")


if __name__ == "__main__":
    main()
