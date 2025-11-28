"""
Data loading and dataset construction for temperature Δ-models.

This package handles loading data from the database and constructing
the snapshot-level feature tables used for training.

Modules:
    loader: DB queries for training data and live inference data
    snapshot_builder: Build snapshot-level feature table from raw data
    splits: Train/test splitting utilities with temporal awareness
"""
