#!/bin/bash
# Back up the second-dataset run directory the same way as the other research runs:
# (1) encrypted restic to Hetzner, (2) plain rsync mirror on the external disk (no --delete).
# Never deletes anything; nice/ionice; no secrets printed. Credentials come from the
# existing env file the nightly research backup uses.
set -uo pipefail; set +x
SRC=${T25_DS2_BACKUP_SRC:-$HOME/t25-dataset2}
ST=$HOME/.local/state/t25-dataset2-backup; mkdir -p "$ST"; L=$ST/backup.log
exec 9>"$ST/lock"; flock -w 600 9 || { echo "$(date -Is) busy" >>"$L"; exit 1; }
EXC=(--exclude=venv --exclude=.venv --exclude=__pycache__ --exclude=pytest_site)
rc=0
( set -a; . "$HOME/.config/research_engine_restic_hz.env"; set +a
  restic cat config >/dev/null 2>>"$L" || { echo "$(date -Is) UNREACHABLE hetzner" >>"$L"; exit 2; }
  nice -n 19 ionice -c3 restic backup -q --limit-upload 15000 --tag research-engine --tag t25-dataset2 \
    "${EXC[@]}" "$SRC" >>"$L" 2>&1 && echo "$(date -Is) ok restic-hetzner" >>"$L" \
    || { echo "$(date -Is) FAIL restic-hetzner" >>"$L"; exit 1; }
) || rc=$?
M=/mnt/homelab-external/backups/research-engine
if mountpoint -q /mnt/homelab-external; then
  nice -n 19 ionice -c3 rsync -a "${EXC[@]}" "$SRC" "$M/" >>"$L" 2>&1 \
    && echo "$(date -Is) ok rsync $SRC" >>"$L" || { echo "$(date -Is) FAIL rsync" >>"$L"; rc=1; }
else echo "$(date -Is) FAIL external disk not mounted" >>"$L"; rc=1; fi
exit $rc
