#!/bin/bash
# Pilot stage of the #25 steering meta-search on engine's RTX 3060.
#
# NOT started automatically. Start it by hand only after the seed jobs have finished
# (seeds 1+2 PID 747584, queued seed-0 rerun waiter PID 1597299 and its run-seed0.sh).
# The guard below refuses to start while any of them is alive, so a stray launch cannot
# share the GPU with the seed runs.
#
# Start command (nice + idle io, like the seed runs):
#   nohup nice -n 19 ionice -c3 bash ~/t25-metasearch/repo/uv/engine/run_metasearch_pilot.sh \
#       >> ~/t25-metasearch/run.log 2>&1 < /dev/null &
#
# Needs: ~/t25-metasearch/repo = a checkout of branch exp/steering-metasearch, and the
# venv + shared files of the seed runs under ~/t25-multiseed (read only here).
# Resumable: start it again with the same arguments and it continues from the per-trial
# checkpoints. Writes only under $ROOT. Deletes nothing. Touches no other process.
# Contains no credentials.
set -euo pipefail

ROOT="${ROOT:-$HOME/t25-metasearch}"
SEEDS_ROOT="${SEEDS_ROOT:-$HOME/t25-multiseed}"
PRIOR_SEED="${PRIOR_SEED:-1}"                    # which finished seed's generators are the priors
SHARED="${SHARED:-$SEEDS_ROOT/ckpt-seeds/shared}" # fixed seed-0 surrogate, corpus, vocab
GEN_DIR="${GEN_DIR:-$SEEDS_ROOT/ckpt-seeds}"
PY="${PY:-$SEEDS_ROOT/repo/venv/bin/python}"
REPO="${REPO:-$ROOT/repo}"
RUN="${RUN:-$ROOT/run-pilot-prior-seed${PRIOR_SEED}}"
DRY_RUN="${DRY_RUN:-0}"

# Pilot schedule (steps = optimiser updates per trained trial, cumulative).
N_GUIDANCE="${N_GUIDANCE:-16}"
N_REINVENT="${N_REINVENT:-24}"
N_SELFTRAIN="${N_SELFTRAIN:-12}"
RUNGS="${RUNGS:-100:1000,300:2000,900:3000}"
CONFIRM_N="${CONFIRM_N:-10000}"

say() { echo "[$(date -Is)] $*"; }

busy=""
for pid in 747584 1597299; do
  if kill -0 "$pid" 2>/dev/null; then busy="$busy $pid"; fi
done
if pgrep -f 'run-seed0.sh|colab_run_seeds.py' >/dev/null 2>&1; then busy="$busy seed-jobs"; fi
if [ -n "$busy" ] && [ "${ALLOW_BUSY_GPU:-0}" != "1" ]; then
  say "refusing to start: seed jobs still running ($busy). They own the GPU."
  exit 3
fi

for f in "$SHARED/corpus.csv" "$SHARED/uvvis_predictor.joblib" "$SHARED/vocab.json" \
         "$SHARED/uvvis_dataset.csv" "$GEN_DIR/generator_plain_seed${PRIOR_SEED}.pt" \
         "$GEN_DIR/generator_steered_seed${PRIOR_SEED}.pt"; do
  [ -s "$f" ] || { say "missing input: $f"; exit 4; }
done
[ -x "$PY" ] || { say "python not found: $PY"; exit 4; }
[ -d "$REPO/uv" ] || { say "no repo checkout at $REPO"; exit 4; }

mkdir -p "$RUN"
cd "$REPO"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
say "host=$(hostname) head=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
say "prior seed=$PRIOR_SEED run dir=$RUN rungs=$RUNGS"

ARGS=(--root "$RUN"
      --plain "$GEN_DIR/generator_plain_seed${PRIOR_SEED}.pt"
      --steered "$GEN_DIR/generator_steered_seed${PRIOR_SEED}.pt"
      --vocab "$SHARED/vocab.json" --corpus "$SHARED/corpus.csv"
      --predictor "$SHARED/uvvis_predictor.joblib" --uvvis-data "$SHARED/uvvis_dataset.csv"
      --n-guidance "$N_GUIDANCE" --n-reinvent "$N_REINVENT" --n-selftrain "$N_SELFTRAIN"
      --rungs "$RUNGS" --eta 3 --n-confirm 3 --confirm-n-eval "$CONFIRM_N")

if [ "$DRY_RUN" = "1" ]; then
  say "DRY RUN. would run:"
  echo "$PY -u -m uv.metasearch_cost ${ARGS[*]} --out $RUN/cost_probe.json"
  echo "$PY -u -m uv.metasearch ${ARGS[*]}"
  exit 0
fi

# 1. one short timing probe on the real device, written next to the results
if [ ! -s "$RUN/cost_probe.json" ]; then
  say "timing probe"
  "$PY" -u -m uv.metasearch_cost "${ARGS[@]}" --out "$RUN/cost_probe.json"
fi
# 2. the pilot (resumes by default)
say "starting meta-search"
exec "$PY" -u -m uv.metasearch "${ARGS[@]}"
