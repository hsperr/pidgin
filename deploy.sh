#!/usr/bin/env bash
# Ship the bid desk to bridge.localgeek.jp. The nginx site and the systemd
# unit live in ~/code/infra.
#
#   ./sync_models.sh && ./deploy.sh               # newest snapshots from bridge_new
#   ./deploy.sh                                   # every model in models/models.json
set -euo pipefail

HOST="root@DEPLOY_HOST_IP"
APP="/opt/bridge/app"
VENV="/opt/bridge/venv"

cd "$(dirname "$0")"
for manifest in models/models.json models/play_models.json; do
  python3 -c "import json,os,sys; [open(os.path.join('models', m['file'])).close() for m in json.load(open(sys.argv[1]))]" "$manifest" \
    || { echo "$manifest names a missing file"; exit 1; }
done

echo "==> syncing code + models to $HOST:$APP"
ssh "$HOST" "mkdir -p $APP"
rsync -az --delete \
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' \
  ./ "$HOST:$APP/"

echo "==> installing deps and restarting"
ssh "$HOST" "VENV='$VENV' APP='$APP' bash -s" <<'REMOTE'
set -euo pipefail
id bridge >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin bridge
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip -q
fi
"$VENV/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
"$VENV/bin/pip" install -q -r "$APP/requirements.txt"

chown -R bridge:bridge /opt/bridge
chmod 750 /opt/bridge
systemctl restart bridge
for i in $(seq 1 30); do curl -sf localhost:3500/api/state -o /dev/null && break; sleep 1; done
systemctl is-active bridge
curl -sf localhost:3500/api/state -o /dev/null && echo "bid desk api ok"
curl -sf localhost:3500/api/play/state -o /dev/null && echo "play desk api ok"
REMOTE

echo "==> done: https://bridge.localgeek.jp"
