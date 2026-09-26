"""Circuit simulator package (WEB_CONTRACT §4): STL element as a state-space circuit element
inside a small MNA transient solver, deterministic (BE/TRAP) and stochastic (Eq. 2 event
increments).  See docs/CIRCUIT_SIMULATOR.md.

``run_circuit(payload, progress)`` is the compute entry point registered as kind "circuit" (the benches of
``benches.py`` and, with ``bench: "custom"``, user-drawn netlists: ``custom.py``, WEB_CONTRACT §6).
The heavy numerical modules (numba, engine) are imported lazily inside that call, so importing
this package (e.g. for ``BENCH_DEFAULTS``) does not load numba.
"""
from __future__ import annotations

from .benches import BENCH_DEFAULTS, BENCH_INFO, CAPS, SOLVER_DEFAULTS, STOCHASTIC_DEFAULTS


def run_circuit(payload: dict, progress=None) -> dict:
    from .runner import run_circuit as _run
    return _run(payload, progress)


__all__ = ["run_circuit", "BENCH_DEFAULTS", "BENCH_INFO", "SOLVER_DEFAULTS", "STOCHASTIC_DEFAULTS", "CAPS"]
