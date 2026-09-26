#!/usr/bin/env bash
# Offline ClaimForge demo. Does not contact OpenAlex, arXiv, or Semantic Scholar.
set -euo pipefail
cd "$(dirname "$0")/.."
if command -v python >/dev/null 2>&1; then
  exec python -m claimforge demo "$@"
fi
exec python3 -m claimforge demo "$@"
