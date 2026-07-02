#!/usr/bin/env bash
# Install (or update) the looper systemd service. Idempotent; needs sudo.
set -euo pipefail

UNIT_SRC="$(cd "$(dirname "$0")" && pwd)/looper.service"

sudo cp "$UNIT_SRC" /etc/systemd/system/looper.service
sudo systemctl daemon-reload
sudo systemctl enable --now looper

echo
systemctl status looper --no-pager --lines=5
