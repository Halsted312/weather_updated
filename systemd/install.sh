#!/bin/bash
# Install Kalshi Weather systemd service
# Run with: sudo ./install.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Installing Kalshi Weather systemd service..."
echo "Project directory: $PROJECT_DIR"

# Copy service file
sudo cp "$SCRIPT_DIR/kalshi-weather.service" /etc/systemd/system/

# Update WorkingDirectory in service file to use actual project path
sudo sed -i "s|WorkingDirectory=.*|WorkingDirectory=$PROJECT_DIR|" /etc/systemd/system/kalshi-weather.service

# Reload systemd
sudo systemctl daemon-reload

# Enable service (starts on boot)
sudo systemctl enable kalshi-weather.service

echo ""
echo "Installation complete!"
echo ""
echo "Commands:"
echo "  sudo systemctl start kalshi-weather    # Start the services"
echo "  sudo systemctl stop kalshi-weather     # Stop the services"
echo "  sudo systemctl status kalshi-weather   # Check status"
echo "  sudo systemctl restart kalshi-weather  # Restart services"
echo "  journalctl -u kalshi-weather -f        # View logs"
echo ""
echo "The service will automatically start on boot."
echo ""
echo "To start now:"
echo "  sudo systemctl start kalshi-weather"
