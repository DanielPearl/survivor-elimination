#!/usr/bin/env bash
# One-shot deploy / redeploy on the Digital Ocean droplet.
#
# Usage (as root):
#   cd /root && git clone <repo-url> survivor-elimination
#   cd survivor-elimination && bash deploy/deploy.sh

set -euo pipefail

REPO_DIR="/root/survivor-elimination"
cd "$REPO_DIR"

if [ -d .git ]; then
  echo "[deploy] git pull"
  git pull --ff-only
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "[deploy] wrote .env from .env.example — edit before starting"
fi

# Initial train + watchlist so the dashboard has data before the
# live monitor takes over.
echo "[deploy] training model — first run takes ~30s"
python scripts/run_daily_train.py --offline

# systemd units.
cp deploy/survivor-elimination-monitor.service /etc/systemd/system/
cp deploy/survivor-elimination-train.service /etc/systemd/system/
cp deploy/survivor-elimination-train.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now survivor-elimination-monitor.service
systemctl enable --now survivor-elimination-train.timer

echo
echo "[deploy] up — live monitor running, daily retrain timer armed"
echo "[deploy] logs:"
echo "  journalctl -u survivor-elimination-monitor -f"
echo "  journalctl -u survivor-elimination-train -f"
