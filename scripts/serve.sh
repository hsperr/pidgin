#!/usr/bin/env bash
# Run the site and the bot APIs locally: http://localhost:${PORT:-8787}
#
#   scripts/serve.sh                      # needs server/models (scripts/get_models.sh)
#   PORT=8080 PLAY_SEARCH=0 scripts/serve.sh   # faster card play, no PIMC search
set -euo pipefail
cd "$(dirname "$0")/../server"
[ -f models/teams.json ] && ls models/*.pt >/dev/null 2>&1 \
  || { echo "no models in server/models: run scripts/get_models.sh"; exit 1; }
exec python -m emergent.bidserver --port "${PORT:-8787}"
