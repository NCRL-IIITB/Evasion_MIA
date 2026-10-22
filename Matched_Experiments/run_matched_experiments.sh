#!/usr/bin/env bash
# =============================================================================
# run_matched_experiments.sh
# =============================================================================
# Matched-control experiment answering the reviewers' shared concern:
#   "the baseline uses augmentation + early stopping, the adversarial model
#    does not — a consistent training setup is needed".
#
# 2x2 design, one training script, one data split, paired seeds:
#   A std_reg     standard loss, aug + early stopping (clean val AUC)
#   B fgsm_reg    FGSM-AT,       aug + early stopping (clean val AUC)
#   B2            = B's run, model selected by adversarial val AUC (robust ES)
#   C std_noreg   standard loss, no aug, no early stopping, 35 epochs
#   D fgsm_noreg  FGSM-AT,       no aug, no early stopping, 35 epochs
# For every victim: FGSM evasion sweep, threshold MIAs on all checkpoints
# (gap-matched analysis), the paper's shadow-model MIA (+BCVA) and AFMIA (+BCVA).
#
# Everything is written under Matched_Experiments/results/ — no original file
# or result is touched.  Re-running resumes: finished steps are skipped.
#
# Usage
#   bash Matched_Experiments/run_matched_experiments.sh --smoke   # ~15 min pipeline test
#   nohup bash Matched_Experiments/run_matched_experiments.sh > matched_run.log 2>&1 &
#
# Environment overrides (optional)
#   SEEDS="42 43 44"     victim + attack seeds              (default 42 43 44)
#   ATTACK_PARALLEL=2    attack/eval jobs sharing the GPU with training
#   RUN_ROBUST_ES=1      also attack B2 (robust early stopping); 0 to skip
#   TRAIN_WORKERS=10     DataLoader workers for victim training
#   PYTHON=/path/python  interpreter (default: auto-detect a venv with torch)
#   RESULTS_DIR=…  CACHE_DIR=…
# =============================================================================

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$HERE")"

SMOKE=0
for arg in "$@"; do
    case "$arg" in
        --smoke) SMOKE=1 ;;
        -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $arg (use --smoke or --help)"; exit 2 ;;
    esac
done

if (( SMOKE )); then
    SEEDS="${SEEDS:-42}"
    RESULTS_DIR="${RESULTS_DIR:-$HERE/results_smoke}"
    SMOKE_FLAG="--smoke"
else
    SEEDS="${SEEDS:-42 43 44}"
    RESULTS_DIR="${RESULTS_DIR:-$HERE/results}"
    SMOKE_FLAG=""
fi
ATTACK_PARALLEL="${ATTACK_PARALLEL:-2}"
RUN_ROBUST_ES="${RUN_ROBUST_ES:-1}"
TRAIN_WORKERS="${TRAIN_WORKERS:-10}"
CACHE_DIR="${CACHE_DIR:-$HERE/cache}"

LOGS="$RESULTS_DIR/logs"
MARK="$RESULTS_DIR/.markers"
VICTIMS="$RESULTS_DIR/victims"
mkdir -p "$LOGS" "$MARK" "$VICTIMS" "$RESULTS_DIR/eval" "$RESULTS_DIR/attacks" "$RESULTS_DIR/splits"
PIPELINE_LOG="$LOGS/pipeline.log"

# Only one instance per results directory (two copies would race on the same files)
exec 9>"$RESULTS_DIR/.lock"
if ! flock -n 9; then
    echo "ERROR: another run_matched_experiments.sh is already using $RESULTS_DIR"
    echo "       (check with: pgrep -af run_matched_experiments)"
    exit 1
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PIPELINE_LOG"; }

trap 'log "Interrupted — stopping all jobs"; trap - INT TERM; kill 0; exit 130' INT TERM

# ── Python interpreter ───────────────────────────────────────────────────────
PY=""
if [[ -n "${PYTHON:-}" ]]; then
    PY="$PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PY="$VIRTUAL_ENV/bin/python"
else
    for cand in "$PROJECT_ROOT/mia_env/bin/python" \
                "$PROJECT_ROOT/../Attacks-on-ML/mia_env/bin/python" \
                "$(command -v python3 || true)"; do
        if [[ -n "$cand" && -x "$cand" ]] && "$cand" -c "import torch" 2>/dev/null; then
            PY="$cand"; break
        fi
    done
fi
if [[ -z "$PY" ]] || ! "$PY" -c "import torch, torchvision, sklearn, scipy, matplotlib" 2>/dev/null; then
    echo "ERROR: no Python with torch/torchvision/sklearn/scipy/matplotlib found. Set PYTHON=…"
    exit 1
fi

# ── Pre-flight ───────────────────────────────────────────────────────────────
log "======================================================================"
log " Matched-control experiment  $( ((SMOKE)) && echo '[SMOKE TEST]')"
log "   python   : $PY"
log "   seeds    : $SEEDS"
log "   parallel : $ATTACK_PARALLEL attack/eval job(s) alongside training"
log "   robustES : $RUN_ROBUST_ES"
log "   results  : $RESULTS_DIR"
log "   cache    : $CACHE_DIR"
log "======================================================================"

if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    log "ERROR: CUDA not available to $PY"; exit 1
fi
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null \
    | while read -r l; do log "   GPU      : $l"; done

free_gb() { df -Pk "$1" | awk 'NR==2 {printf "%d", $4/1048576}'; }
need_gb=4
[[ -f "$CACHE_DIR/cache_info.json" ]] || need_gb=10
if (( $(free_gb "$HERE") < need_gb )); then
    log "ERROR: only $(free_gb "$HERE") GB free on $(df -P "$HERE" | awk 'NR==2{print $6}'); need ≥ ${need_gb} GB"
    exit 1
fi
log "   disk free: $(free_gb "$HERE") GB"

# Fresh attempt: forget previous failures (completed steps stay completed)
rm -f "$MARK"/*.failed "$MARK/train_lane.finished"

"$PY" - "$RESULTS_DIR/run_config.json" "$SEEDS" "$ATTACK_PARALLEL" "$RUN_ROBUST_ES" "$SMOKE" <<'EOF'
import json, os, subprocess, sys, datetime, torch, torchvision, sklearn
path, seeds, par, rob, smoke = sys.argv[1:]
try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                     cwd=os.path.dirname(path)).strip()
except Exception:
    commit = None
cfg = {"started": datetime.datetime.now().isoformat(timespec="seconds"),
       "seeds": [int(s) for s in seeds.split()], "attack_parallel": int(par),
       "robust_es": rob == "1", "smoke": smoke == "1", "git_commit": commit,
       "torch": torch.__version__, "torchvision": torchvision.__version__,
       "sklearn": sklearn.__version__, "gpu": torch.cuda.get_device_name(0)}
hist = path.replace(".json", "_history.jsonl")
with open(hist, "a") as f:
    f.write(json.dumps(cfg) + "\n")
json.dump(cfg, open(path, "w"), indent=2)
EOF

# ── Stage 1: image cache (one-off, ~5.6 GB) ──────────────────────────────────
log "STAGE 1: image cache"
if ! "$PY" "$HERE/build_image_cache.py" --cache_dir "$CACHE_DIR" >> "$LOGS/build_cache.log" 2>&1; then
    log "ERROR: cache build failed — see $LOGS/build_cache.log"; exit 1
fi
log "   cache ready ($(tail -1 "$LOGS/build_cache.log"))"

# ── Run plan ─────────────────────────────────────────────────────────────────
# Training order per seed puts the decisive NEW controls (B, C) first.
TRAIN_CELLS=(fgsm_reg std_noreg std_reg fgsm_noreg)

cell_training() { case "$1" in std_*) echo standard ;; fgsm_*) echo fgsm ;; esac; }
cell_regime()   { case "$1" in *_noreg) echo noreg ;; *) echo reg ;; esac; }
# B2 is produced by B's training run
train_source()  { echo "${1/fgsm_reg_robustES_/fgsm_reg_}"; }

PLANNED_RUNS=()
for seed in $SEEDS; do
    for cell in "${TRAIN_CELLS[@]}"; do
        PLANNED_RUNS+=("${cell}_seed${seed}")
        if [[ "$cell" == fgsm_reg && "$RUN_ROBUST_ES" == 1 ]]; then
            PLANNED_RUNS+=("fgsm_reg_robustES_seed${seed}")
        fi
    done
done

# ── Training lane (background, one victim at a time) ─────────────────────────
train_lane() {
    for seed in $SEEDS; do
        for cell in "${TRAIN_CELLS[@]}"; do
            local run="${cell}_seed${seed}"
            if [[ -f "$MARK/train_${run}.done" ]]; then
                log "train $run: already done"; continue
            fi
            local extra=()
            if [[ "$cell" == fgsm_reg && "$RUN_ROBUST_ES" == 1 ]]; then
                extra=(--robust_es_out_dir "$VICTIMS/fgsm_reg_robustES_seed${seed}")
            fi
            log "train $run: START"
            if "$PY" "$HERE/train_matched_victim.py" \
                    --training "$(cell_training "$cell")" --regime "$(cell_regime "$cell")" \
                    --seed "$seed" --out_dir "$VICTIMS/$run" --cache_dir "$CACHE_DIR" \
                    --num_workers "$TRAIN_WORKERS" "${extra[@]}" $SMOKE_FLAG \
                    > "$LOGS/train_${run}.log" 2>&1; then
                touch "$MARK/train_${run}.done"
                log "train $run: DONE ($(grep -m1 '^\[DONE\]' "$LOGS/train_${run}.log" | cut -c1-120))"
            else
                touch "$MARK/train_${run}.failed"
                log "train $run: FAILED — see $LOGS/train_${run}.log"
            fi
        done
    done
    touch "$MARK/train_lane.finished"
}

# ── Job pool for evaluation / attacks ────────────────────────────────────────
JOB_PIDS=()
RUNNING=0
count_running() {
    local alive=() p
    RUNNING=0
    for p in "${JOB_PIDS[@]}"; do
        if kill -0 "$p" 2>/dev/null; then alive+=("$p"); RUNNING=$((RUNNING + 1)); fi
    done
    JOB_PIDS=("${alive[@]}")
}

LAST_AGG=$(date +%s)
maybe_aggregate() {   # refresh summary/REPORT.md at most every 30 min
    if (( $(date +%s) - LAST_AGG > 1800 )); then
        "$PY" "$HERE/aggregate_results.py" --results_dir "$RESULTS_DIR" >> "$LOGS/aggregate.log" 2>&1
        LAST_AGG=$(date +%s)
    fi
}

launch_job() {   # launch_job <name> <logfile> <cmd …>
    local name="$1" logf="$2"; shift 2
    if [[ -f "$MARK/$name.done" ]]; then return 0; fi
    while :; do
        count_running
        (( RUNNING < ATTACK_PARALLEL )) && break
        sleep 15
        maybe_aggregate
    done
    log "job $name: START"
    (
        if "$@" > "$logf" 2>&1; then
            touch "$MARK/$name.done"; log "job $name: DONE"
        else
            touch "$MARK/$name.failed"; log "job $name: FAILED — see $logf"
        fi
    ) &
    JOB_PIDS+=($!)
}

# ── Stage 2+3: train in the background, attack each victim as soon as it exists
log "STAGE 2: training lane started (${#PLANNED_RUNS[@]} victims planned)"
train_lane &
TRAIN_PID=$!

for run in "${PLANNED_RUNS[@]}"; do
    src="$(train_source "$run")"
    while [[ ! -f "$MARK/train_${src}.done" ]]; do
        [[ -f "$MARK/train_${src}.failed" || -f "$MARK/train_lane.finished" ]] && break
        sleep 30
        maybe_aggregate
    done
    if [[ ! -f "$MARK/train_${src}.done" || ! -f "$VICTIMS/$run/model.pth" ]]; then
        log "skip $run: victim not available"; continue
    fi
    seed="${run##*_seed}"
    common=(--victim_dir "$VICTIMS/$run" --seed "$seed" --cache_dir "$CACHE_DIR")
    # longest job first for better packing
    launch_job "afmia_${run}" "$LOGS/afmia_${run}.log" \
        "$PY" "$HERE/run_shadow_attacks.py" --attack afmia "${common[@]}" \
        --out_dir "$RESULTS_DIR/attacks/afmia/$run" --splits_dir "$RESULTS_DIR/splits" $SMOKE_FLAG
    launch_job "shadow_mia_${run}" "$LOGS/shadow_mia_${run}.log" \
        "$PY" "$HERE/run_shadow_attacks.py" --attack shadow_mia "${common[@]}" \
        --out_dir "$RESULTS_DIR/attacks/shadow_mia/$run" --splits_dir "$RESULTS_DIR/splits" $SMOKE_FLAG
    launch_job "eval_${run}" "$LOGS/eval_${run}.log" \
        "$PY" "$HERE/evaluate_victim.py" "${common[@]}" \
        --out_dir "$RESULTS_DIR/eval/$run" $SMOKE_FLAG
done

log "STAGE 3: all jobs dispatched — waiting for completion"
while :; do
    count_running
    train_alive=0; kill -0 "$TRAIN_PID" 2>/dev/null && train_alive=1
    (( RUNNING == 0 && train_alive == 0 )) && break
    sleep 60
    maybe_aggregate
done
wait

# ── Stage 4: aggregate ───────────────────────────────────────────────────────
log "STAGE 4: aggregating"
"$PY" "$HERE/aggregate_results.py" --results_dir "$RESULTS_DIR" 2>&1 | tee -a "$LOGS/aggregate.log"

failed=( $(ls "$MARK"/*.failed 2>/dev/null) )
if (( ${#failed[@]} )); then
    log "FINISHED WITH ${#failed[@]} FAILED STEP(S):"
    for f in "${failed[@]}"; do log "   - $(basename "$f" .failed)  (log: $LOGS/$(basename "$f" .failed).log)"; done
    log "Fix the cause and re-run this script; completed steps are skipped."
    exit 1
fi
log "ALL DONE — report: $RESULTS_DIR/summary/REPORT.md"
