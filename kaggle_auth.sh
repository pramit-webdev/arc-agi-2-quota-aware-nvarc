#!/bin/bash
# Safe Kaggle auth loader - never prints the secret.
# Usage: source ./kaggle_auth.sh && kaggle competitions list ...
# Maps KAGGLE_API_KEY from .env -> KAGGLE_API_TOKEN (new CLI format)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env"
  set +a
fi
if [ -n "${KAGGLE_API_KEY:-}" ] && [ -z "${KAGGLE_API_TOKEN:-}" ]; then
  export KAGGLE_API_TOKEN="$KAGGLE_API_KEY"
fi
# Do NOT echo the token. Only confirm presence.
if [ -z "${KAGGLE_API_TOKEN:-}" ]; then
  echo "ERROR: No Kaggle token found. Check .env (KAGGLE_API_KEY) - not printed for safety." >&2
  return 1 2>/dev/null || exit 1
else
  echo "Kaggle auth loaded (token length ${#KAGGLE_API_TOKEN}, hidden)."
fi
