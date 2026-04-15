from typing import Any
from dataclasses import dataclass, field

from rapmat.storage.base import StructureStore


@dataclass
class AppState:
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
        self.runs_cache = self.store.list_runs()
        self.cache_dirty = False

    def refresh_runs_if_needed(self) -> None:
        if self.cache_dirty:
            self.refresh_runs()

    def refresh_studies(self) -> None:
        self.studies_cache = self.store.list_studies()
        self.studies_cache_dirty = False

    def refresh_studies_if_needed(self) -> None:
        if self.studies_cache_dirty:
            self.refresh_studies()

    def invalidate_runs(self) -> None:
        self.cache_dirty = True
        self.active_run_meta = None

    def invalidate_studies(self) -> None:
        self.studies_cache_dirty = True

    def invalidate(self) -> None:
        self.invalidate_runs()
        self.invalidate_studies()

    def reconnect(self, new_store: "StructureStore") -> None:
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
