"""CPU inference scheduling shared by the embedder and the reranker.

torch runs one model call on all cores, so two concurrent calls only slow each other
down. Every model call therefore passes one gate. Interactive work (a query's
embedding, reranking) goes before batch work (ingest embedding): a waiting query is
let in at the next micro-batch boundary instead of after a whole document.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


class InferenceGate:
    """One model call at a time; interactive callers overtake waiting batch callers."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._busy = False
        self._interactive_waiting = 0

    @contextmanager
    def interactive(self, timeout_s: float | None = None) -> Iterator[bool]:
        """Enter ahead of batch work. Yields False when *timeout_s* passed first."""
        with self._cond:
            self._interactive_waiting += 1
            try:
                acquired = self._cond.wait_for(lambda: not self._busy, timeout=timeout_s)
            finally:
                self._interactive_waiting -= 1
            if acquired:
                self._busy = True
            else:
                self._cond.notify_all()  # batch callers may go now that we stopped waiting
        try:
            yield acquired
        finally:
            if acquired:
                self._release()

    @contextmanager
    def batch(self) -> Iterator[None]:
        """Enter when the gate is free and no interactive caller is waiting."""
        with self._cond:
            self._cond.wait_for(lambda: not self._busy and self._interactive_waiting == 0)
            self._busy = True
        try:
            yield
        finally:
            self._release()

    def _release(self) -> None:
        with self._cond:
            self._busy = False
            self._cond.notify_all()


GATE = InferenceGate()


def available_cpus() -> int | None:
    """CPUs this process may use: affinity, capped by the cgroup quota (containers).

    torch's default is the host's core count, which oversubscribes a CPU-capped
    container (e.g. 32 threads under ``--cpus 4``).
    """
    affinity = getattr(os, "sched_getaffinity", None)  # Linux only
    count = len(affinity(0)) if affinity is not None else (os.cpu_count() or 0)
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            count = min(count, max(1, -(-int(quota) // int(period))))
    except (OSError, ValueError):
        pass
    return count or None


def configure_threads(requested: int) -> int | None:
    """Set torch's (process-wide) thread count, and docling's to the same value.

    *requested* 0 derives it: on Linux from :func:`available_cpus` (torch would count the
    host's cores, oversubscribing a CPU-capped container); elsewhere torch's own default
    is kept (on Apple silicon it counts the performance cores only). Docling otherwise
    sets torch to its own default (4) during the first PDF conversion, silently changing
    the embedder's speed. Returns the count in effect, or None without torch.
    """
    try:
        import torch
    except ImportError:
        return None
    threads = requested if requested > 0 else None
    if threads is None and sys.platform.startswith("linux"):
        threads = available_cpus()
    if threads:
        torch.set_num_threads(threads)
    effective = int(torch.get_num_threads())
    os.environ.setdefault("DOCLING_NUM_THREADS", str(effective))
    logger.info("Inference threads: %d", effective)
    return effective
