#!/bin/bash
# Watch the queued LB33 notebook version, validate its submission output, and
# submit it to the ARC Prize 2026 - ARC-AGI-2 code competition.
#
# Why this exists: L4x4 queue times are long (often many hours).  The kernel
# version is already queued (commit).  Once it reaches COMPLETE we (1) download
# the output, (2) validate submission.json, and (3) submit the exact version
# with the required "-f submission.json" (omitting it produced a 400 Bad
# Request from CreateCodeSubmission).
#
# Safe to leave running: it makes at most ONE submission, and only after the
# output validates.  Logs to /tmp/lb33_autosubmit.log.
set -u

ROOT="/home/pramit/Desktop/Projects/ARC Prize 2026 - ARC-AGI-2"
KERNEL="pramitdas/arc-2026-agi2-nvarc-lb33-v1"
COMP="arc-prize-2026-arc-agi-2"
VERSION="7"
OUTDIR="/tmp/lb33_v${VERSION}_out"
LOG="/tmp/lb33_autosubmit.log"
MAX_POLLS=960          # 960 * 180s ~= 48 hours (L4x4 queues have been 24h+)
POLL_SECONDS=180

cd "$ROOT" || exit 1
# shellcheck disable=SC1091
source ./kaggle_auth.sh >/dev/null 2>&1

log() { echo "[$(date --iso-8601=seconds)] $*" | tee -a "$LOG"; }

log "watcher start: kernel=$KERNEL version=$VERSION"

status=""
for i in $(seq 1 "$MAX_POLLS"); do
  status="$(kaggle kernels status "$KERNEL" 2>/dev/null | tr -d '\r')"
  log "poll $i: $status"
  case "$status" in
    *COMPLETE*) break ;;
    *ERROR*|*CANCELLED*|*CANCELED*)
      log "kernel did not complete cleanly ($status); stopping without submitting."
      exit 3 ;;
  esac
  sleep "$POLL_SECONDS"
done

case "$status" in
  *COMPLETE*) : ;;
  *) log "timed out waiting for completion (last=$status); stopping."; exit 4 ;;
esac

# Download + validate the produced submission.json before spending the daily
# submission.
rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
kaggle kernels output "$KERNEL" -p "$OUTDIR" >>"$LOG" 2>&1
log "downloaded output to $OUTDIR"

if ! python3 - "$OUTDIR/submission.json" >>"$LOG" 2>&1 <<'PY'
import json, sys
path = sys.argv[1]
d = json.load(open(path))
assert isinstance(d, dict) and d, "submission is not a non-empty dict"
bad = 0
for k, v in d.items():
    assert isinstance(v, list) and len(v) == 2, f"{k}: expected 2 test entries"
    for entry in v:
        for a in ("attempt_1", "attempt_2"):
            g = entry.get(a)
            assert isinstance(g, list) and g and isinstance(g[0], list), f"{k}:{a} bad grid"
print(f"VALIDATED: {len(d)} task keys, all entries well-formed")
PY
then
  log "submission.json failed validation; NOT submitting."
  exit 5
fi

log "submitting version $VERSION ..."
kaggle competitions submit "$COMP" -k "$KERNEL" -v "$VERSION" \
  -f submission.json -m "nvarc lb33 + robust selector, 12h budget (v$VERSION)" \
  >>"$LOG" 2>&1
log "submit command returned $?"

sleep 15
log "current submissions:"
kaggle competitions submissions "$COMP" >>"$LOG" 2>&1
log "watcher done."