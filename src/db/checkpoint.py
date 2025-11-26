"""
Checkpoint management for resumable ingestion pipelines.

Provides functions to track progress and enable resume on crash/restart.
"""

import logging
from datetime import date, datetime
from typing import Optional

from sqlalchemy import select, update, and_
from sqlalchemy.orm import Session

from src.db.models import IngestCheckpoint

logger = logging.getLogger(__name__)


def get_or_create_checkpoint(
    session: Session,
    pipeline_name: str,
    city: Optional[str] = None,
) -> IngestCheckpoint:
    """
    Get existing running checkpoint or create a new one.

    Args:
        session: SQLAlchemy session
        pipeline_name: Name of the pipeline (e.g., 'kalshi_markets', 'vc_minutes')
        city: Optional city filter (None for all-city runs)

    Returns:
        IngestCheckpoint instance (existing or new)
    """
    # Look for existing running checkpoint
    query = select(IngestCheckpoint).where(
        and_(
            IngestCheckpoint.pipeline_name == pipeline_name,
            IngestCheckpoint.city == city if city else IngestCheckpoint.city.is_(None),
            IngestCheckpoint.status == "running",
        )
    )
    checkpoint = session.execute(query).scalar_one_or_none()

    if checkpoint:
        logger.info(
            f"Resuming checkpoint: {pipeline_name}/{city or 'all'} "
            f"from {checkpoint.last_processed_date or 'start'} "
            f"({checkpoint.total_processed} processed)"
        )
        return checkpoint

    # Create new checkpoint
    checkpoint = IngestCheckpoint(
        pipeline_name=pipeline_name,
        city=city,
        status="running",
    )
    session.add(checkpoint)
    session.flush()  # Get the ID
    logger.info(f"Created new checkpoint: {pipeline_name}/{city or 'all'}")
    return checkpoint


def update_checkpoint(
    session: Session,
    checkpoint_id: int,
    last_date: Optional[date] = None,
    last_cursor: Optional[str] = None,
    last_ticker: Optional[str] = None,
    processed_count: int = 0,
    error: Optional[str] = None,
) -> None:
    """
    Update checkpoint progress.

    Args:
        session: SQLAlchemy session
        checkpoint_id: ID of the checkpoint to update
        last_date: Last successfully processed date
        last_cursor: Last pagination cursor
        last_ticker: Last processed ticker
        processed_count: Number of items processed in this batch
        error: Error message if any
    """
    updates = {"updated_at": datetime.utcnow()}

    if last_date is not None:
        updates["last_processed_date"] = last_date
    if last_cursor is not None:
        updates["last_processed_cursor"] = last_cursor
    if last_ticker is not None:
        updates["last_processed_ticker"] = last_ticker
    if processed_count > 0:
        # Use raw SQL for increment to avoid race conditions
        session.execute(
            update(IngestCheckpoint)
            .where(IngestCheckpoint.id == checkpoint_id)
            .values(
                total_processed=IngestCheckpoint.total_processed + processed_count,
                **updates,
            )
        )
    elif error:
        session.execute(
            update(IngestCheckpoint)
            .where(IngestCheckpoint.id == checkpoint_id)
            .values(
                error_count=IngestCheckpoint.error_count + 1,
                last_error=error,
                **updates,
            )
        )
    else:
        session.execute(
            update(IngestCheckpoint)
            .where(IngestCheckpoint.id == checkpoint_id)
            .values(**updates)
        )

    session.flush()


def complete_checkpoint(
    session: Session,
    checkpoint_id: int,
    status: str = "completed",
) -> None:
    """
    Mark checkpoint as completed or failed.

    Args:
        session: SQLAlchemy session
        checkpoint_id: ID of the checkpoint
        status: Final status ('completed' or 'failed')
    """
    session.execute(
        update(IngestCheckpoint)
        .where(IngestCheckpoint.id == checkpoint_id)
        .values(
            status=status,
            completed_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
    )
    session.flush()
    logger.info(f"Checkpoint {checkpoint_id} marked as {status}")


def get_last_checkpoint_date(
    session: Session,
    pipeline_name: str,
    city: Optional[str] = None,
) -> Optional[date]:
    """
    Get the last successfully processed date for a pipeline/city.

    Useful for determining where to resume from.

    Args:
        session: SQLAlchemy session
        pipeline_name: Name of the pipeline
        city: Optional city filter

    Returns:
        Last processed date or None if no checkpoint exists
    """
    query = select(IngestCheckpoint.last_processed_date).where(
        and_(
            IngestCheckpoint.pipeline_name == pipeline_name,
            IngestCheckpoint.city == city if city else IngestCheckpoint.city.is_(None),
        )
    ).order_by(IngestCheckpoint.updated_at.desc())

    result = session.execute(query).scalar_one_or_none()
    return result


def reset_stale_checkpoints(
    session: Session,
    max_age_hours: int = 24,
) -> int:
    """
    Reset checkpoints that have been 'running' for too long (likely crashed).

    Args:
        session: SQLAlchemy session
        max_age_hours: Consider checkpoints stale after this many hours

    Returns:
        Number of checkpoints reset
    """
    from datetime import timedelta

    cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)

    result = session.execute(
        update(IngestCheckpoint)
        .where(
            and_(
                IngestCheckpoint.status == "running",
                IngestCheckpoint.updated_at < cutoff,
            )
        )
        .values(
            status="failed",
            last_error=f"Reset: stale for >{max_age_hours} hours",
            updated_at=datetime.utcnow(),
        )
    )

    count = result.rowcount
    if count > 0:
        logger.warning(f"Reset {count} stale checkpoints")
    return count
