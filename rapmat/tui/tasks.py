"""Background task infrastructure for the Rapmat TUI."""

import threading
from dataclasses import dataclass, field
from typing import Callable

import urwid

# ------------------------------------------------------------------ #
#  Thread-safe progress state
# ------------------------------------------------------------------ #


@dataclass
class TaskProgress:
    """Shared state written by the worker thread, read by the poll callback."""

    total: int = 0
    current: int = 0
    message: str = ""
    finished: bool = False
    error: str | None = None
    cancelled: bool = False
    log_lines: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, current: int, total: int = 0, message: str = "") -> None:
        """Report progress from the worker thread."""
        with self._lock:
            self.current = current
            if total:
                self.total = total
            self.message = message

    def log(self, message: str) -> None:
        """Append a log line from the worker thread and echo to physical log file."""
        with self._lock:
            self.log_lines.append(message)
            
        try:
            from datetime import datetime
            from rapmat.config import APP_DATA_DIR

            log_dir = APP_DATA_DIR / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "background.log"

            with open(log_file, "a", encoding="utf-8") as f:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{ts}] {message}\n")
        except Exception:
            pass

    def finish(self) -> None:
        """Mark the task as successfully completed."""
        with self._lock:
            self.finished = True

    def fail(self, error: str) -> None:
        """Mark the task as failed with an error message."""
        with self._lock:
            self.error = error
            self.finished = True

    def drain_logs(self) -> list[str]:
        """Pop and return all pending log lines (called from UI thread)."""
        with self._lock:
            lines = list(self.log_lines)
            self.log_lines.clear()
        return lines


# ------------------------------------------------------------------ #
#  Background task runner
# ------------------------------------------------------------------ #


class BackgroundTask:
    """Run a callable in a background thread and poll its progress.

    Parameters
    ----------
    fn:
        Worker function.  Receives a single ``TaskProgress`` argument.
        Must check ``progress.cancelled`` periodically and return early
        if set.
    loop:
        The ``urwid.MainLoop`` instance.  Used to schedule poll alarms.
    on_progress:
        Called on the UI thread with ``(current: int, total: int, message: str)`` each
        poll cycle while the task is running.
    on_log:
        Called on the UI thread with each new log line drained from the
        progress queue.
    on_complete:
        Called on the UI thread with no arguments when the task finishes
        successfully.
    on_error:
        Called on the UI thread with ``(error: str)`` when the task fails.
    poll_interval:
        Seconds between poll alarms.
    """

    _POLL_INTERVAL = 0.3

    def __init__(
        self,
        fn: Callable[[TaskProgress], None],
        loop: urwid.MainLoop,
        on_progress: Callable[[int, int, str], None] | None = None,
        on_log: Callable[[str], None] | None = None,
        on_complete: Callable[[], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        poll_interval: float = _POLL_INTERVAL,
    ) -> None:
        self._fn = fn
        self._loop = loop
        self._on_progress = on_progress
        self._on_log = on_log
        self._on_complete = on_complete
        self._on_error = on_error
        self._poll_interval = poll_interval
        self._progress = TaskProgress()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Spawn the worker thread and schedule the first poll alarm."""
        self._progress = TaskProgress()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._loop.set_alarm_in(self._poll_interval, self._poll)

    def cancel(self) -> None:
        """Request cancellation.  The worker must honour ``progress.cancelled``."""
        self._progress.cancelled = True

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        """Worker thread entry point."""
        try:
            self._fn(self._progress)
            if not self._progress.finished:
                self._progress.finish()
        except KeyboardInterrupt:
            self._progress.fail("Cancelled by user.")
        except Exception as exc:
            self._progress.fail(str(exc))

    def _poll(self, loop: urwid.MainLoop, data) -> None:
        """Poll callback executed on the UI thread."""
        p = self._progress

        if self._on_log:
            for line in p.drain_logs():
                self._on_log(line)

        if self._on_progress:
            self._on_progress(p.current, p.total, p.message)

        if p.finished:
            if p.error:
                if self._on_error:
                    self._on_error(p.error)
            else:
                if self._on_complete:
                    self._on_complete()
            return

        loop.set_alarm_in(self._poll_interval, self._poll)
