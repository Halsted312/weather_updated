# Legacy Folder Notes

This directory contains **archived/legacy ingestion scripts** kept for reference.

Guidelines for agents:

- Treat files here as **read‑only by default**.
- Do **not** refactor, rename, or delete anything in `legacy/` unless the user explicitly asks.
- If a task touches legacy logic, prefer migrating/copying the needed behavior into the current pipeline under `scripts/` or `src/`, then note what was borrowed.
- When you find duplicated or dead code in `legacy/`, report it as context but do not change it.

