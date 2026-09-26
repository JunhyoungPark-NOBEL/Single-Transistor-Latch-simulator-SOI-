"""Progress protocol shared by all compute functions.

Every compute entry point has the signature ``run(payload: dict, progress: Progress) -> dict``.
``progress(fraction, message)`` reports 0..1 completion and raises ``JobCancelled`` when the user
cancelled the job, so long loops must call it regularly (at least every ~0.5 s of work).
"""
from __future__ import annotations

from typing import Callable, Protocol


class JobCancelled(Exception):
    """Raised from a progress callback when the job was cancelled."""


class Progress(Protocol):
    def __call__(self, fraction: float, message: str = "") -> None: ...


def null_progress(fraction: float, message: str = "") -> None:  # for tests / direct calls
    return None


ProgressFn = Callable[[float, str], None]
