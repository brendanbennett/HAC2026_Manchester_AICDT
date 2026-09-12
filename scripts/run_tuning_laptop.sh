#!/usr/bin/env bash
# Launch scripts/tune_genetic_hyperparams.py as several parallel worker processes on this
# laptop, all against one shared Optuna study, sized to run for about 8 hours.
#
#   ./scripts/run_tuning_laptop.sh
#
# Why processes, not scripts/tune_genetic_hyperparams.py's own --n-jobs: --n-jobs threads
# a single process, and the GA's own evaluation (hac26.shapes.mesh_curves_convex) is
# numpy/Python-dominated CPU work with nothing that releases the GIL for long, so threads do
# not give real parallelism here. Separate OS processes do; Optuna is built for exactly this
# -- every worker independently calls study.optimize() against the same --storage URL,
# picking its next trial via TPE from whatever history already exists, so N processes reading
# and writing the same SQLite file is the intended way to parallelise across cores, not a
# workaround.
#
# Sizing: this machine has 10 CPU cores (`sysctl -n hw.ncpu` / `os.cpu_count()`).
# N_WORKERS=8 leaves 2 free so the laptop stays usable; each worker is single-threaded
# (OMP_NUM_THREADS=1 etc. below) so 8 workers do not oversubscribe 10 cores against each
# other. Each worker runs its own --timeout of TIMEOUT_HOURS hours; since workers proceed
# independently, that is the total wall clock for the run, not TIMEOUT_HOURS/N_WORKERS --
# an in-flight trial finishes rather than being cut off, so real wall clock can run a little
# past this.
#
# Effective cost, measured on this laptop against actual shape-library convex hulls:
# ~9ms per lightcurve evaluation, so one trial costs roughly
#   n_shapes * population_size * (n_generations + 1) * 9ms
# SEARCH_SPACE in scripts/tune_genetic_hyperparams.py bounds population_size and
# n_generations specifically to keep a single trial's worst case around 20-25 minutes (not
# the 1-2 hours the full historically-explored range would allow), so no one unlucky trial
# on any one worker eats a large share of that worker's 8-hour budget. With N_SHAPES=8 and
# those bounds, expected trial cost averages ~8 minutes, so 8 workers over 8 hours should
# complete on the order of 400-500 trials in total -- a real number depends on where Optuna's
# TPE sampler actually spends its time, which is the point of letting --timeout govern this
# rather than guessing --n-trials.
#
# STL files are NOT written during any of this: scripts/tune_genetic_hyperparams.py's
# run_ga_once discards every candidate mesh after scoring its Dice, keeping only the
# hyperparameters and the score. Only best_params.json and trials.csv (written once, at the
# end) and the SQLite study file touch disk during the run.
#
# FORWARD_MODEL=exact (a remote GPU box, not this laptop): each worker gets its own CUDA
# context on the SAME device by default, and ExactForward.raw_curves takes one mesh per call
# (no batching across the GA's population) -- so unlike the convex case, more worker
# processes here does not add real throughput past however many candidates the GPU can
# actually run at once, and instead adds VRAM pressure and context-switch overhead on a box
# that may be shared with other users. Set N_WORKERS=1 or 2 for FORWARD_MODEL=exact, not 8.
set -euo pipefail
cd "$(dirname "$0")/.."

N_WORKERS=${N_WORKERS:-8}
TIMEOUT_HOURS=${TIMEOUT_HOURS:-8}
N_SHAPES=${N_SHAPES:-8}
SHAPES_DIR=${SHAPES_DIR:-dataset/generated/shapes}
SEED=${SEED:-42}
STORAGE=${STORAGE:-sqlite:///runs/genetic_tuning_laptop.db}
OUTPUT_DIR=${OUTPUT_DIR:-results/genetic/hyperparameter_tuning}
FORWARD_MODEL=${FORWARD_MODEL:-convex}
CALIBRATION=${CALIBRATION:-models/instrument_calibration.pt}
EXACT_DEVICE=${EXACT_DEVICE:-cpu}
EXACT_BACKEND=${EXACT_BACKEND:-}

TIMEOUT_SECONDS=$(python3 -c "print(int(${TIMEOUT_HOURS} * 3600))")

mkdir -p "$(dirname "${STORAGE#sqlite:///}")" "$OUTPUT_DIR"

echo "[run_tuning_laptop] $N_WORKERS workers, timeout ${TIMEOUT_HOURS}h each, " \
    "n_shapes=$N_SHAPES, forward_model=$FORWARD_MODEL, storage=$STORAGE"
echo "[run_tuning_laptop] all workers share --seed $SEED, so they evaluate the SAME test " \
    "shapes -- required for one valid shared study, not an arbitrary choice"

# Create the study ONCE, sequentially, before any worker starts. optuna.create_study()'s
# schema setup (RDBStorage's create_all) is not safe against several processes hitting a
# brand-new sqlite file at the same moment -- confirmed by testing this launcher: concurrent
# first-time creation raises "table studies already exists" in whichever workers lose the
# race. Pre-creating means every worker below just loads an already-existing study instead.
.venv/bin/python -c "
import optuna
optuna.create_study(study_name='genetic_ga_tuning', storage='$STORAGE', load_if_exists=True,
                    direction='maximize')
print('[run_tuning_laptop] study ready')
"

PIDS=()
for i in $(seq 1 "$N_WORKERS"); do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
    .venv/bin/python scripts/tune_genetic_hyperparams.py \
      --shapes-dir "$SHAPES_DIR" \
      --n-shapes "$N_SHAPES" \
      --seed "$SEED" \
      --storage "$STORAGE" \
      --output-dir "$OUTPUT_DIR" \
      --forward-model "$FORWARD_MODEL" \
      --calibration "$CALIBRATION" \
      --exact-device "$EXACT_DEVICE" \
      ${EXACT_BACKEND:+--exact-backend "$EXACT_BACKEND"} \
      --timeout "$TIMEOUT_SECONDS" \
      > "$OUTPUT_DIR/worker_$i.log" 2>&1 &
  worker_pid=$!
  PIDS+=("$worker_pid")
  echo "[run_tuning_laptop] started worker $i, pid $worker_pid, log: $OUTPUT_DIR/worker_$i.log"
done

echo "[run_tuning_laptop] waiting on ${#PIDS[@]} workers (this blocks for up to "\
    "~${TIMEOUT_HOURS}h)..."
wait "${PIDS[@]}"

echo "[run_tuning_laptop] all workers finished; results in $OUTPUT_DIR"
echo "[run_tuning_laptop] monitor live with: .venv/bin/optuna-dashboard $STORAGE"
