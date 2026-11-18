#!/bin/bash
# Round 2 of the #25 steering meta-search on engine's RTX 3060 (diversity-aware REINVENT
# with checkpoint selection, then 3-seed confirmation of the winner if there is one).
#
# Start (nice + idle io):
#   nohup nice -n 19 ionice -c3 bash ~/t25-metasearch/repo/uv/engine/run_metasearch_round2.sh \
#       >> ~/t25-metasearch/run-round2.log 2>&1 < /dev/null &
#
# Refuses to start while any other #25 job holds the GPU. Resumable: start it again and it
# continues from the per-trial checkpoints and the per-snapshot results file. Writes only
# under $ROOT. Deletes nothing. Signals no process. Contains no credentials.
set -euo pipefail

ROOT="${ROOT:-$HOME/t25-metasearch}"
SEEDS_ROOT="${SEEDS_ROOT:-$HOME/t25-multiseed}"
SHARED="${SHARED:-$SEEDS_ROOT/ckpt-seeds/shared}"
PRIOR_SEED="${PRIOR_SEED:-1}"                       # search runs on this seed's priors
GEN1="${GEN1:-$SEEDS_ROOT/ckpt-seeds}"              # seeds 1 and 2
GEN0="${GEN0:-$SEEDS_ROOT/ckpt-seed0-3060}"         # seed 0 (RTX 3060 rerun)
PY="${PY:-$SEEDS_ROOT/repo/venv/bin/python}"
REPO="${REPO:-$ROOT/repo}"
RUN="${RUN:-$ROOT/run-round2-prior-seed${PRIOR_SEED}}"
N_LOCAL="${N_LOCAL:-40}"
N_GLOBAL="${N_GLOBAL:-55}"
RUNGS="${RUNGS:-300,600,1200}"
PILOT_HOLDOUT="$ROOT/run-pilot-prior-seed1/holdout_predictor.joblib"

say() { echo "[$(date -Is)] $*"; }

# Another #25 GPU job must not be running (never touches anything else on the box).
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
  say "refusing to start: a process already uses the GPU:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
  exit 3
fi
if pgrep -f 'uv[./]metasearch|uv/run.py|train_generator|colab_run_seeds|run-seed0' >/dev/null 2>&1; then
  say "refusing to start: another #25 job is running"; exit 3
fi

for f in "$SHARED/corpus.csv" "$SHARED/uvvis_predictor.joblib" "$SHARED/vocab.json" \
         "$SHARED/uvvis_dataset.csv" "$GEN1/generator_plain_seed${PRIOR_SEED}.pt" \
         "$GEN1/generator_steered_seed${PRIOR_SEED}.pt"; do
  [ -s "$f" ] || { say "missing input: $f"; exit 4; }
done
[ -x "$PY" ] || { say "python not found: $PY"; exit 4; }
mkdir -p "$RUN"
# reuse the exact holdout predictor of the pilot (same file, same metrics)
if [ -s "$PILOT_HOLDOUT" ] && [ ! -e "$RUN/holdout_predictor.joblib" ]; then
  cp "$PILOT_HOLDOUT" "$RUN/holdout_predictor.joblib"
fi
cd "$REPO"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
say "host=$(hostname) head=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD) run=$RUN"

"$PY" -m uv.metasearch2 --mode search --root "$RUN" \
  --plain "$GEN1/generator_plain_seed${PRIOR_SEED}.pt" \
  --steered "$GEN1/generator_steered_seed${PRIOR_SEED}.pt" \
  --vocab "$SHARED/vocab.json" --corpus "$SHARED/corpus.csv" \
  --predictor "$SHARED/uvvis_predictor.joblib" --uvvis-data "$SHARED/uvvis_dataset.csv" \
  --n-local "$N_LOCAL" --n-global "$N_GLOBAL" --rungs "$RUNGS"

# Winner (if any) -> 3 prior seeds, 10,000 fresh samples each, mean +- sd.
WIN=$("$PY" - "$RUN/results_round2.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
w = d.get("winner")
print("" if not w else json.dumps({"spec": w["spec"], "step": w["step"]}))
PYEOF
)
if [ -z "$WIN" ]; then
  say "no configuration passed every gate and the holdout check; see $RUN/results_round2.md (frontier)"
  exit 0
fi
STEPS=$(echo "$WIN" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["step"])')
echo "$WIN" | "$PY" -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["spec"]))' > "$RUN/winner_spec.json"
say "winner at step $STEPS: $(cat "$RUN/winner_spec.json")"
"$PY" -m uv.metasearch2 --mode seeds --root "$RUN/seeds" \
  --vocab "$SHARED/vocab.json" --corpus "$SHARED/corpus.csv" \
  --predictor "$SHARED/uvvis_predictor.joblib" --uvvis-data "$SHARED/uvvis_dataset.csv" \
  --spec-json "$RUN/winner_spec.json" --steps "$STEPS" \
  --seed-priors "0:$GEN0/generator_plain_seed0.pt:$GEN0/generator_steered_seed0.pt,1:$GEN1/generator_plain_seed1.pt:$GEN1/generator_steered_seed1.pt,2:$GEN1/generator_plain_seed2.pt:$GEN1/generator_steered_seed2.pt"
say "done: $RUN/seeds/seeds_result.md"
