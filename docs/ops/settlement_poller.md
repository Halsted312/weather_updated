# Settlement Poller Ops

The CLI/CF6 poller (`scripts/poll_settlements.py`) now runs continuously via
`init/systemd/kalshi-settlement-poller.service`. This page documents how to keep
the loop healthy and how to publish the requested daily row-count summaries.

## Deploy / Manage the Service

```bash
sudo cp init/systemd/kalshi-settlement-poller.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now kalshi-settlement-poller.service
```

Useful commands:

| Command | Purpose |
| --- | --- |
| `systemctl status kalshi-settlement-poller.service` | One-shot health check |
| `journalctl -u kalshi-settlement-poller.service -n 200` | Recent runs/logs |
| `journalctl -u kalshi-settlement-poller.service -f` | Follow live loop |

## Daily Roll-up

Use `scripts/settlement_rollup.py` to summarise how many CLI/CF6/VC/GHCND rows
landed per city. Example (yesterday only):

```bash
python scripts/settlement_rollup.py --days 1 --output results/settlement_rollup_$(date +%Y%m%d).csv
```

Schedule the command via cron/systemd timer (after the final daily poll) and
drop the CSV into your monitoring artefacts bucket. The script also logs the
counts so journalctl already has the summary line.

## Troubleshooting Checklist

1. **Nothing inserting?** Run the roll-up for the last 3 days. If CLI counts are
   zero but CF6 exists, the NOAA CLI endpoint probably failed. Manually rerun
   the poller without `--loop` to backfill.
2. **Service flaps?** `journalctl -u` will show stack traces; common culprits
   are expired API keys or Postgres restarts. Restart with
   `sudo systemctl restart kalshi-settlement-poller.service` after fixing the
   root cause.
3. **Materialized view stale?** After large backfills rerun
   `python ingest/load_kalshi_data.py --refresh-grid` to keep the
   `wx.minute_obs_1m` grid aligned with the new settlements.
