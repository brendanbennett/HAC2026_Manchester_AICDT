# SUBMISSION_depth_lpd_flow

One of the submitted methods, kept whole and kept apart. It is the `depth-field` branch as a
directory: its own `hac26/` package, its own `scripts/`, its own trained networks under
`models/`, and its own README.md and REPRODUCE.md.

The method is a learned primal-dual solver written as a conditional flow
(`hac26/solvers/lpd_flow.py`). A convex inversion gives a first body and the flow corrects it,
several times over from different noise; one of those answers is then chosen. The curves do
not determine the body -- distinct shapes produce identical lightcurves -- so the solver
returns a spread of bodies that fit rather than a single regression, and the choice between
them is made by measuring. README.md here is the full account.

## The submitted bodies

`results/submission/Asteroid04.stl` to `Asteroid10.stl`, one per scored model, with
`provenance.json` recording where each one came from and its sha256. Every file is one
watertight component of positive volume in the challenge pose, checked by
`scripts/check_submission.py`.

They are not all from the same solver, and provenance.json is how to tell which is which.
Where the flow produced a body it is the flow's; where it produced none -- it refuses to
answer when no draw renders -- the Gauss-Newton refinement or the convex answer stands
instead. `scripts/assemble_submission.py` is what puts them in place, preferring the newest
run and falling back per model, and it copies nothing until every chosen file has passed the
submission check.

## Why a copy and not a merge

The two trees share no history. `git merge-base main depth-field` returns nothing, so there is
no common ancestor to merge against.

More to the point, they cannot share code even where the names match. This method began from
main and then changed a great deal, so both have a package called `hac26`, with modules,
classes and functions of the same names that no longer do the same things: about 2550 lines
apart across 23 files, with `solvers/gauss_newton.py`, `solvers/convex_direct.py` and
`scoring/official.py` on this side only, and a `submission.py` on the other side only. They
are not two versions of one package. They are two packages that share a name.

## Why running it cannot pick up the wrong one

Every script here puts its own directory at the front of the import path before importing
anything:

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

so `import hac26` finds the package beside the script and not the one at the repository root,
whatever the working directory. `reproduce.sh` opens with `cd "$(dirname "$0")"` for the same
reason. Checked from the repository root, where the other `hac26` sits in plain view, `hac26`
still resolves here and `hac26.solvers.gauss_newton` imports -- a module the root package does
not contain.

Install into an environment made from inside this directory: `make venv` here builds one under
this directory. Installing both packages into one environment would put two different `hac26`
distributions in one `site-packages`, and install order rather than anything visible would
decide which answers an import.

## Running it

Start at `REPRODUCE.md`, or run `./reproduce.sh` from inside this directory for a summary of
the stages and what each costs. Everything needed is here, including the trained networks;
only the organisers' released data has to be fetched.
