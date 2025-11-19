# Systemd Jobs (Settlement Poller)

The `scripts/poll_settlements.py` runner can now execute in a loop (30‑minute cadence by default). Use the included service unit to keep the process alive on boot.

## Installation

```bash
sudo cp /home/halsted/Documents/python/kalshi_weather/init/systemd/kalshi-settlement-poller.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now kalshi-settlement-poller.service
```

The service calls `scripts/run_settlement_poller.sh`, which loads `.env`, runs the poller for all cities, refreshes CF6 prelims, and sleeps for 1,800 seconds between iterations. Restart logic is handled by systemd (`Restart=always`).

## Monitoring

```bash
sudo systemctl status kalshi-settlement-poller.service
journalctl -u kalshi-settlement-poller.service -f
```

The poller uses `--days-back 3` so each refresh backfills three recent days, ensuring we never miss delayed CLI releases. Adjust cadence via `scripts/run_settlement_poller.sh` or override `--interval-seconds` inside the service file before reloading systemd if you need a different schedule.
