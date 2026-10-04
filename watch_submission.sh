#!/bin/bash
# Poll the v7 competition submission until Kaggle finishes scoring it, logging
# the score.  The private rerun on the hidden test set can take ~12h plus the
# L4x4 queue, so this runs for up to ~33h.
set -u

ROOT="/home/pramit/Desktop/Projects/ARC Prize 2026 - ARC-AGI-2"
COMP="arc-prize-2026-arc-agi-2"
REF="56831159"          # submission created from notebook v7 on 2026-10-04
LOG="/tmp/lb33_submission_monitor.log"

cd "$ROOT" || exit 1
# shellcheck disable=SC1091
source ./kaggle_auth.sh >/dev/null 2>&1

log() { echo "[$(date --iso-8601=seconds)] $*" | tee -a "$LOG"; }

log "monitor start (competition=$COMP ref=$REF)"
for i in $(seq 1 400); do          # 400 * 300s ~= 33 hours
  row="$(kaggle competitions submissions "$COMP" 2>/dev/null | grep "$REF" || true)"
  log "poll $i: ${row:-<ref $REF not listed>}"
  if echo "$row" | grep -q 'COMPLETE'; then
    log "submission $REF is COMPLETE (score shown in the row above)."
    break
  fi
  sleep 300
done
log "monitor done."