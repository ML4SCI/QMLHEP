#!/usr/bin/env bash
# Reproduces every reported top-tagging result, one end-to-end run per seed:
#   preprocessing -> classical KAN -> QKAN (KAN warm start) -> QKAN (random-init ablation)
#   -> Random Forest baseline, then the untrained Chebyshev/sine/random comparison and the
#   aggregate metrics table.
#
# Usage: scripts/reproduce_results.sh [--full-dataset] [--force] [--foreground]
#   (default)        hyperparams.py's regime: invariant-mass cut + n_subsets disjoint
#                    partitions; seed % n_subsets selects the subset
#   --full-dataset   entire dataset: no mass cut, n_subsets=1, train/val/test kept separate
#   --force          re-run every stage even if its checkpoint already exists (default: skip)
#   --foreground     don't self-background via nohup (log goes to the terminal)
# Seeds default to 10 11 12 13 14; override with e.g. SEEDS="10 11" scripts/reproduce_results.sh.
#
# The regime is enforced end to end: the same flag is passed to preprocessing and to every
# training step, and each regime writes to its own directories
# (see src/utils/workspace.py: outputs/top/<cut>/<full|n{N}_subset{k}>/seed_<seed>/).
set -e

SCRIPT_PATH="$(readlink -f "$0")"

cd "$(dirname "$SCRIPT_PATH")/.."

FULL_ARGS=()
FORCE_ARGS=()
REGIME="partitioned"
FOREGROUND=0
for arg in "$@"; do
    case "$arg" in
        --full-dataset) FULL_ARGS=(--full-dataset); REGIME="full" ;;
        --force)        FORCE_ARGS=(--force) ;;
        --foreground)   FOREGROUND=1 ;;
        *)
            echo "Unknown argument: $arg (usage: $0 [--full-dataset] [--force] [--foreground])" >&2
            exit 2
            ;;
    esac
done
SEEDS="${SEEDS:-10 11 12 13 14}"

if [ "$FOREGROUND" -eq 0 ] && [ -z "$PIPELINE_BACKGROUNDED" ]; then
    LOG_DIR="$(PYTHONPATH=. python3 -c 'from src.utils.workspace import get_config; print(get_config("top", 0)["pipeline_logs_dir"])')"
    mkdir -p "$LOG_DIR"
    PIPELINE_LOG="${LOG_DIR}/pipeline_${REGIME}_$(date +%Y%m%d_%H%M%S).log"
    export PIPELINE_BACKGROUNDED=1
    nohup "$SCRIPT_PATH" "$@" > "$PIPELINE_LOG" 2>&1 &
    echo "Pipeline launched in background (PID $!)."
    echo "Combined log: $PIPELINE_LOG"
    echo "Per-seed logs are written under the run directory's logs/ (see workspace.get_config)."
    exit 0
fi

echo "[1/4] Preprocessing top-tagging data (regime: ${REGIME})..."
# Builds the canonical partition for this regime (each regime has its own cache directory);
# training scripts only ever select from a prebuilt partition.
python3 scripts/run_preprocessing.py "${FULL_ARGS[@]}" "${FORCE_ARGS[@]}"

echo "[2/4] Training per seed: ${SEEDS}"
for SEED in $SEEDS; do
    echo "========================================================================"
    echo "STARTING PIPELINE RUN FOR SEED: ${SEED} ($(date))"
    echo "========================================================================"

    echo "[2a] Classical KAN pipeline (seed=${SEED})..."
    python3 scripts/train_kan.py --seed ${SEED} "${FULL_ARGS[@]}" "${FORCE_ARGS[@]}"

    echo "[2b] QKAN from the KAN warm start (seed=${SEED})..."
    python3 scripts/train_qkan.py --seed ${SEED} --task top "${FULL_ARGS[@]}" "${FORCE_ARGS[@]}"

    echo "[2c] QKAN random-init ablation (seed=${SEED})..."
    python3 scripts/train_qkan.py --seed ${SEED} --task top --random_init "${FULL_ARGS[@]}" "${FORCE_ARGS[@]}"

    echo "[2d] Random Forest baseline (seed=${SEED})..."
    python3 scripts/train_rf.py --seed ${SEED} --task top "${FULL_ARGS[@]}" "${FORCE_ARGS[@]}"

    echo "========================================================================"
    echo "FINISHED PIPELINE RUN FOR SEED: ${SEED} ($(date))"
    echo "========================================================================"
done

echo "[3/4] Untrained warm-start comparison (Chebyshev vs. sine vs. random)..."
python3 scripts/eval_sine_baseline.py --seeds ${SEEDS} "${FULL_ARGS[@]}"

echo "[4/4] Refreshing the aggregate metrics and eval-data tables..."
python3 scripts/collect_metrics.py --task top
python3 scripts/collect_eval_data.py --task top
