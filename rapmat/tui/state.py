"""Shared TUI application state."""

from dataclasses import dataclass, field
from typing import Any
from rapmat.storage.base import StructureStore


@dataclass
class AppState:
    """Central state container for the TUI application."""

    store: "StructureStore"
    db_url: str = ""
    loop: Any = None
    status_bar: Any = None

    active_run: str | None = None
    active_study: str | None = None
    active_run_meta: dict | None = None

    runs_cache: list[dict] = field(default_factory=list)
    studies_cache: list[dict] = field(default_factory=list)
    cache_dirty: bool = True
    studies_cache_dirty: bool = True

    def refresh_runs(self) -> None:
        """Reload runs from the store and reset the dirty flag."""
        self.runs_cache = self.store.list_runs()
        self.cache_dirty = False

    def refresh_runs_if_needed(self) -> None:
        """Reload only when the cache has been invalidated."""
        if self.cache_dirty:
            self.refresh_runs()

    def refresh_studies(self) -> None:
        """Reload studies from the store and reset the dirty flag."""
        self.studies_cache = self.store.list_studies()
        self.studies_cache_dirty = False

    def refresh_studies_if_needed(self) -> None:
        """Reload only when the studies cache has been invalidated."""
        if self.studies_cache_dirty:
            self.refresh_studies()

    def invalidate_runs(self) -> None:
        """Mark only the runs cache as stale."""
        self.cache_dirty = True
        self.active_run_meta = None

    def invalidate_studies(self) -> None:
        """Mark only the studies cache as stale."""
        self.studies_cache_dirty = True

    def invalidate(self) -> None:
        """Mark all caches as stale so the next screen refresh re-fetches."""
        self.invalidate_runs()
        self.invalidate_studies()

    def reconnect(self, new_store: "StructureStore") -> None:
        """Replace the active store, close the old one, and flush caches."""
        old = self.store
        self.store = new_store
        self.db_url = getattr(new_store, "_db_url", "")
        self.active_run = None
        self.active_study = None
        self.active_run_meta = None
        self.invalidate()
        try:
            old.close()
        except Exception:
            pass
