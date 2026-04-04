from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter


class WallClockProfiler:
    """Collect coarse wall-clock timings grouped by named buckets.

    Buckets prefixed with ``vulkan_`` count toward the derived Vulkan subtotal.
    Buckets prefixed with ``cpu_`` count toward the explicit CPU subtotal.
    All other buckets are preserved as raw wall-clock markers in the snapshot so
    the CLI timing report can retain orchestration wait points such as queue
    handshakes and thread boundaries.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._durations: defaultdict[str, float] = defaultdict(float)
        self._counts: defaultdict[str, int] = defaultdict(int)

    def add_duration(self, bucket: str, duration_s: float) -> None:
        """Accumulate a duration that was measured outside `measure()`."""
        self._durations[bucket] += float(duration_s)

    def add_count(self, bucket: str, count: int = 1) -> None:
        """Accumulate an integer counter associated with a bucket."""
        self._counts[bucket] += int(count)

    @contextmanager
    def measure(self, bucket: str):
        start = perf_counter()
        try:
            yield
        finally:
            self._durations[bucket] += perf_counter() - start
            self._counts[bucket] += 1

    def raw_snapshot(self) -> dict[str, float | int]:
        """Return the currently tracked durations and counts without derived totals."""
        snapshot: dict[str, float | int] = dict(self._durations)
        for bucket, count in self._counts.items():
            snapshot[f"{bucket}__count"] = count
        return snapshot

    def snapshot(self, *, total_s: float) -> dict[str, float | int]:
        durations = dict(self._durations)
        counts = dict(self._counts)
        vulkan_total_s = sum(
            duration for bucket, duration in durations.items() if bucket.startswith("vulkan_")
        )
        tracked_cpu_total_s = sum(
            duration for bucket, duration in durations.items() if bucket.startswith("cpu_")
        )
        cpu_total_s = max(total_s - vulkan_total_s, 0.0)
        cpu_untracked_s = max(cpu_total_s - tracked_cpu_total_s, 0.0)

        snapshot: dict[str, float | int] = {
            "total_s": total_s,
            "cpu_total_s": cpu_total_s,
            "tracked_cpu_total_s": tracked_cpu_total_s,
            "cpu_untracked_s": cpu_untracked_s,
            "vulkan_total_s": vulkan_total_s,
        }
        snapshot.update(durations)
        for bucket, count in counts.items():
            snapshot[f"{bucket}__count"] = count
        return snapshot
