#!/usr/bin/env bash
# Download the released Pidgin models into server/models/ (about 140 MB).
#
#   scripts/get_models.sh                 # every released file
#   PIDGIN_HF_REPO=user/pidgin scripts/get_models.sh
set -euo pipefail
REPO="${PIDGIN_HF_REPO:-hsperr/pidgin}"
cd "$(dirname "$0")/.."
command -v hf >/dev/null || python -m pip install -q "huggingface_hub[cli]"
hf download "$REPO" --repo-type model --local-dir server/models \
  --include "*.pt" "*.npz" "brl_LICENSE"
ls -lh server/models | grep -E '\.(pt|npz)$'
