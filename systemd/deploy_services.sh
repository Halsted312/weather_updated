#!/bin/bash
# Deploy systemd services for weather data ingestion
# Run with: sudo ./systemd/deploy_services.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Deploying Weather Data Ingestion Services ==="

# Copy service files
echo "Copying service files to /etc/systemd/system/..."
cp "$SCRIPT_DIR/vc-live-daemon.service" /etc/systemd/system/
cp "$SCRIPT_DIR/kalshi-ws-recorder.service" /etc/systemd/system/
cp "$SCRIPT_DIR/kalshi-candle-poller.service" /etc/systemd/system/

# Reload systemd
echo "Reloading systemd daemon..."
systemctl daemon-reload

# Enable services
echo "Enabling services..."
systemctl enable vc-live-daemon.service
systemctl enable kalshi-ws-recorder.service
systemctl enable kalshi-candle-poller.service

# Start services
echo "Starting services..."
systemctl start vc-live-daemon.service
systemctl start kalshi-ws-recorder.service
systemctl start kalshi-candle-poller.service

# Check status
echo ""
echo "=== Service Status ==="
systemctl status vc-live-daemon.service --no-pager || true
echo ""
systemctl status kalshi-ws-recorder.service --no-pager || true
echo ""
systemctl status kalshi-candle-poller.service --no-pager || true

echo ""
echo "=== Deployment Complete ==="
echo "View logs with:"
echo "  journalctl -u vc-live-daemon -f"
echo "  journalctl -u kalshi-ws-recorder -f"
echo "  journalctl -u kalshi-candle-poller -f"
