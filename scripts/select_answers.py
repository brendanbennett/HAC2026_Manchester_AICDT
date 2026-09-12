#!/usr/bin/env python3
"""Decide, per scored model, whether a refined body replaces the convex answer.

    python scripts/select_answers.py --refined results/gn --calibrate results/gn/Asteroid03.json

A refinement is accepted for a model when its number on the geometries held out of its fit,
measured on the body as written, is at most `ratio` times the convex answer's number on the
same geometries, and the file passes the submission check. The held-out geometries are the
whole of the argument: a body fitted on every camera can reach any misfit by shape or by
overfitting, and only a camera the fit never saw separates the two.

The number is whichever functional the refinement was fitted under, and `held_pair` says why
it has to be. A correction fitted under an objective that charges the body's surface as well
as its misfit, judged here on the misfit alone, would be thrown away in favour of a corrugated
body that fits the curves better and looks less like the truth.

The ratio is one by default: a refinement stands where it fits the held-out geometries at
least as well as the convex answer does, which is evidence about the body being decided. It
can be tightened by a measured quantity instead. `--calibrate` reads a public model's
refinement, checks that the refinement moved that body toward its released truth rather than
away from it, and takes the misfit ratio it reached; a refinement is then trusted only where
it beats its convex answer by at least as much. That is one body's margin asked of every
other, so it is offered and not required: a public run that falls short would otherwise stand
every convex answer in the submission, and a convex answer is not a safe default here but a
body known to be missing the concavities the challenge is about. `--ratio` sets the number
directly.

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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_submission import inspect                       # noqa: E402
from hac26.conventions import CYLINDER_R, PUBLIC_MODELS    # noqa: E402
from reconstruct import answer_path                        # noqa: E402

SELECTION_FILE = "results/submission/selection.json"
RATIO_MARGIN = 1.0    # how much of a public model's ratio a scored model has to reproduce.
                      # One asks for the same improvement; the ratio is one measurement of
                      # one body, and asking a secret body to beat it is not warranted.


def held_pair(meta: dict) -> tuple:
    """(the correction's held-out number, the convex answer's) on whichever functional the
    correction was fitted under.

    A correction fitted under a penalised objective must be judged by it. The penalty exists
    because the misfit of a rendered body reports how finely its surface is resolved as well
    as whether its shape is right, so on the misfit alone this gate prefers a corrugated body
    to a shaped one and would throw the better body away. A run that minimised the misfit
    alone reports no objective and is judged on the misfit, which is the same thing for it.

    The objective is a logarithm plus an area, so it is a difference rather than a ratio that
    is meaningful; the two are put on a common footing by exponentiating, under which a ratio
    of exponentials is the ratio of misfits the same body would have at equal area.
    """
    held, base = meta.get("objective_held_export"), meta.get("objective_held_convex")
    if held is not None and base is not None and held == held and base == base:
        return float(np.exp(0.5 * held)), float(np.exp(0.5 * base))
    return meta.get("chi_held_export"), meta.get("chi_held_convex")


def calibrated_ratio(meta: dict) -> float:
    """The largest accepted misfit ratio, from a public model's refinement of known truth.

    A refinement that lowered the misfit while lowering the overlap with the truth is the
    failure this gate exists to catch, so it yields no ratio at all rather than a lenient
    one. So does a refinement of a body whose truth is not released, or one fitted on every
    camera, whose misfit is not a test of anything.
    """
    if not meta.get("held_out"):
        raise SystemExit("the calibrating run held out no geometries, so its misfit ratio "
                         "is not a test of a shape")
    dice, base_dice = meta.get("final_dice"), meta.get("convex_dice")
    if dice is None or base_dice is None or not (dice == dice and base_dice == base_dice):
        raise SystemExit("the calibrating run is of a model whose truth is not released, so "
                         "there is nothing to calibrate against")
    if dice <= base_dice:
        raise SystemExit(f"the calibrating refinement took the overlap from {base_dice:.4f} "
                         f"to {dice:.4f}, so a lower misfit is not evidence of a better "
                         f"shape and the convex answers stand")
    held, convex = held_pair(meta)
    if not held or not convex or not (held < float("inf")):
        raise SystemExit("the calibrating run has no held-out misfit")
    return RATIO_MARGIN * held / convex


def decide(meta: dict, ratio: float) -> tuple:
    """(accept, reason) for one refinement's JSON. A refinement fitted without held-out
    geometries has no honest number and is refused."""
    held, base = held_pair(meta)
    if not meta.get("held_out"):
        return False, "no geometries were held out of the fit"
    if held is None or base is None or not (held < float("inf")):
        return False, "the written body has no held-out misfit"
    if held > ratio * base:
        return False, f"held out {held:.3f} above {ratio:g} x the convex answer's {base:.3f}"
    return True, f"held out {held:.3f} at or below {ratio:g} x the convex answer's {base:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refined", required=True,
                    help="directory of Asteroid<NN>.stl and .json written by "
                         "reconstruct_gn.py or reconstruct_map.py")
    ap.add_argument("--calibrate",
                    help="a public model's refinement JSON from the same run; the accepted "
                         "misfit ratio is the one it reached, and only if it improved the "
                         "overlap with the released truth")
    ap.add_argument("--ratio", type=float,
                    help="the accepted ratio directly, when the calibrating run is not at "
                         "hand; largest held-out misfit of a refinement as a fraction of the "
                         "convex answer's")
    ap.add_argument("--models", nargs="+", type=int,
                    default=[M for M in range(1, 11) if M not in PUBLIC_MODELS])
    ap.add_argument("--into", default=None,
                    help="directory the chosen bodies are written to; by default the one "
                         "answer_path names, which is the submission itself. Two tracks that "
                         "both select into it overwrite each other, so give each its own "
                         "directory and let a person choose between them. A directory other "
                         "than the default is seeded with the convex answer for every model "
                         "first, so what it holds is a complete submission either way")
    a = ap.parse_args()
    if a.calibrate is not None and a.ratio is not None:
        raise SystemExit("give either --calibrate, to measure the ratio on a public model, "
                         "or --ratio to set it, not both")
    calibration = json.loads(Path(a.calibrate).read_text()) if a.calibrate else None
    # With neither, the ratio is one: a refinement is accepted when it fits the geometries held
    # out of its own fit at least as well as the convex answer does. That is evidence about the
    # body being decided. A public model's ratio is a tightening on top of it, and it is one
    # body's margin asked of every other, so it is offered rather than required -- a public run
    # that falls short would otherwise stand every convex answer in the submission.
    ratio = (a.ratio if a.ratio is not None else
             calibrated_ratio(calibration) if calibration is not None else 1.0)
    if not 0.0 < ratio <= 1.0:
        raise SystemExit(f"the accepted ratio {ratio:g} is outside (0, 1]: a refinement that "
                         f"fits the held-out geometries no better than the convex answer is "
                         f"not accepted")
    if calibration is not None:
        print(f"ratio {ratio:.3f}, from model {calibration['model']}, whose refinement took "
              f"the overlap with its released truth from {calibration['convex_dice']:.4f} to "
              f"{calibration['final_dice']:.4f}", flush=True)
    elif a.ratio is None:
        print("ratio 1.000: a refinement stands where it fits the held-out geometries at "
              "least as well as the convex answer. Pass --calibrate to require a public "
              "model's own margin as well.", flush=True)

    into = Path(a.into) if a.into else None
    if into is not None:
        into.mkdir(parents=True, exist_ok=True)
        for M in a.models:
            src = answer_path(M)
            dst = into / Path(src).name
            if not dst.exists():
                shutil.copyfile(src, dst)
        print(f"seeded {into} with the convex answer for {len(a.models)} models; what it "
              f"holds is a complete submission whatever is accepted below", flush=True)
    selection_file = str(into / "selection.json") if into else SELECTION_FILE
    selection = {"ratio": ratio, "refined_dir": a.refined, "into": str(into) if into else None,
                 "calibrated_on": a.calibrate, "models": {}}
    for M in a.models:
        target = str(into / Path(answer_path(M)).name) if into else answer_path(M)
        stl = Path(a.refined) / f"Asteroid{M:02d}.stl"
        js = stl.with_suffix(".json")
        entry = {"answer": "convex"}
        if not (stl.exists() and js.exists()):
            entry["reason"] = "no refinement written"
        else:
            meta = json.loads(js.read_text())
            accept, reason = decide(meta, ratio)
            check = inspect(stl, CYLINDER_R[M])
            if accept and check["fails"]:
                accept, reason = False, "refined file fails the submission check: " + \
                    "; ".join(check["fails"])
            held, base = held_pair(meta)
            entry.update({"reason": reason, "held_convex": base, "held_refined": held,
                          "chi_held_convex": meta.get("chi_held_convex"),
                          "chi_held_export": meta.get("chi_held_export"),
                          "held_out": meta.get("held_out")})
            if accept:
                shutil.copyfile(stl, target)
                entry["answer"] = "refined"
        selection["models"][M] = entry
        print(f"model {M:>2}: {entry['answer']:8s} {entry['reason']}", flush=True)
    Path(selection_file).write_text(json.dumps(selection, indent=2))
    print(f"wrote {selection_file}")


if __name__ == "__main__":
    main()
