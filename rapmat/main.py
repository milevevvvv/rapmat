def app_main() -> None:

    from rapmat.db_config import resolve_store
    from rapmat.tui.app import RapmatApp
    from rapmat.tui.state import AppState

    startup_error: Exception | None = None
    try:
        store = resolve_store()
    except Exception as exc:
        startup_error = exc
        from rapmat.storage.surrealdb_store import SurrealDBStore

        store = SurrealDBStore(db_url="mem://")

    state = AppState(store=store, db_url=getattr(store, "_db_url", ""))
    RapmatApp(state, startup_error=startup_error).run()
