"""Rich progress-bar utilities for evaluation runs.

Mirrors the reference ``aieng.agent_evals.progress`` module.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

T = TypeVar("T")


def create_progress(*, transient: bool = False) -> Progress:
    """Create a standardised Rich progress bar."""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        transient=transient,
        console=Console(force_jupyter=False),
    )


def track_with_progress(
    iterable: Iterable[T],
    *,
    description: str,
    total: int | None = None,
    transient: bool = False,
) -> Iterator[T]:
    """Iterate items while displaying a progress bar."""
    resolved_total: int | None = total
    if resolved_total is None:
        try:
            resolved_total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            resolved_total = None

    with create_progress(transient=transient) as progress:
        task_id = progress.add_task(description, total=resolved_total)
        for item in iterable:
            yield item
            progress.update(task_id, advance=1)


__all__ = ["create_progress", "track_with_progress"]
