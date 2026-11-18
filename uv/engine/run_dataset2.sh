#!/bin/bash
# #25 second-dataset replication on the engine GPU (RTX 3060).
#
# Queues behind the steering meta-search: polls with pgrep and kill -0 only, never signals
# or stops any process, deletes nothing. The GPU is used only after no meta-search or
# multi-seed process has been alive for four consecutive 30 s checks.
# Then: train the steered arm for seeds 0, 1, 2 (three processes in parallel, nice 19),
# evaluate each seed (plain checkpoints reused, see uv/dataset2/run_dataset2.py), write
# the aggregate, and back the results up (encrypted restic to Hetzner plus the plain
# local copy on the external disk). No secrets are read or printed here.
#
# Usage: bash uv/engine/run_dataset2.sh        (log to stdout; run it under nohup)
# Completion signal: the line "DS2 DONE" in the log and aggregate_ds2.json under $ROOT.
set -uo pipefail
HOME_=$HOME
ROOT=${T25_DS2_ROOT:-$HOME_/t25-dataset2/ckpt}
REPO=${T25_DS2_REPO:-$HOME_/t25-dataset2/repo}
PY=${T25_DS2_PY:-$HOME_/t25-multiseed/repo/venv/bin/python}
MAIN_SHARED=${T25_MAIN_SHARED:-$HOME_/t25-multiseed/ckpt-seeds/shared}
PLAIN0=${T25_PLAIN0:-$HOME_/t25-multiseed/ckpt-seed0-3060/generator_plain_seed0.pt}
PLAIN1=${T25_PLAIN1:-$HOME_/t25-multiseed/ckpt-seeds/generator_plain_seed1.pt}
PLAIN2=${T25_PLAIN2:-$HOME_/t25-multiseed/ckpt-seeds/generator_plain_seed2.pt}
say() { echo "[$(date -Is)] $*"; }
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$REPO" || exit 1

busy() { pgrep -f 'uv\.metasearch|run_metasearch|colab_run_seeds|run-seed0' >/dev/null 2>&1; }

if [ -z "${T25_DS2_NO_WAIT:-}" ]; then
  say "waiting for the meta-search / seed jobs to finish (GPU is theirs until then)"
  idle=0
  while [ "$idle" -lt 4 ]; do
    if busy; then idle=0; else idle=$((idle+1)); fi
    sleep 30
  done
fi
say "GPU free; starting. nvidia-smi: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"

"$PY" -m uv.dataset2.run_dataset2 --root "$ROOT" prepare \
  --main-corpus "$MAIN_SHARED/corpus.csv" --main-vocab "$MAIN_SHARED/vocab.json" || { say "prepare failed"; exit 1; }

pids=()
for s in 0 1 2; do
  nice -n 19 ionice -c3 "$PY" -u -m uv.dataset2.run_dataset2 --root "$ROOT" train --seed "$s" \
    > "$ROOT/train_steered_seed$s.log" 2>&1 &
  pids+=($!)
  say "seed $s steered training pid=$! log=$ROOT/train_steered_seed$s.log"
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 0 ] || { say "a training process failed; see the train_steered_seed*.log files"; exit 1; }

for s in 0 1 2; do
  "$PY" -u -m uv.dataset2.run_dataset2 --root "$ROOT" evaluate --seed "$s" \
    --plain-ckpt "0=$PLAIN0" "1=$PLAIN1" "2=$PLAIN2" --main-predictor "$MAIN_SHARED/uvvis_predictor.joblib" \
    > "$ROOT/eval_seed$s.log" 2>&1 || { say "evaluate seed $s failed"; exit 1; }
  say "evaluated seed $s"
done
"$PY" -u -m uv.dataset2.run_dataset2 --root "$ROOT" report --seeds 0 1 2 > "$ROOT/report.log" 2>&1 \
  || { say "report failed"; exit 1; }
bash "$(dirname "$0")/backup_dataset2.sh" || say "backup reported a failure (see its log)"
say "DS2 DONE"
