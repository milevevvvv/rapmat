import urwid
import threading

from dataclasses import dataclass, field
from typing import Callable

# ------------------------------------------------------------------ #
#  Thread-safe progress state
# ------------------------------------------------------------------ #


@dataclass
class TaskProgress:
    total: int = 0
    current: int = 0
    message: str = ""
    finished: bool = False
    error: str | None = None
    cancelled: bool = False
    log_lines: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, current: int, total: int = 0, message: str = "") -> None:
        with self._lock:
            self.current = current
            if total:
                self.total = total
            self.message = message

    def log(self, message: str) -> None:
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
        with self._lock:
            self.finished = True

    def fail(self, error: str) -> None:
        with self._lock:
            self.error = error
            self.finished = True

    def drain_logs(self) -> list[str]:
        with self._lock:
            lines = list(self.log_lines)
            self.log_lines.clear()
        return lines


# ------------------------------------------------------------------ #
#  Background task runner
# ------------------------------------------------------------------ #


class BackgroundTask:
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
        self._progress = TaskProgress()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._loop.set_alarm_in(self._poll_interval, self._poll)

    def cancel(self) -> None:
        self._progress.cancelled = True

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        try:
            self._fn(self._progress)
            if not self._progress.finished:
                self._progress.finish()
        except KeyboardInterrupt:
            self._progress.fail("Cancelled by user.")
        except Exception as exc:
            self._progress.fail(str(exc))

    def _poll(self, loop: urwid.MainLoop, data) -> None:
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
