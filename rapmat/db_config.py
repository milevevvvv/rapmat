import tomllib

from pathlib import Path

from rapmat.storage.base import StructureStore
from rapmat.config import APP_CONFIG_DIR, APP_DATA_DIR

_DB_CONFIG_FILE = APP_CONFIG_DIR / "db.toml"

_SERVER_DEFAULTS: dict[str, str | None] = {
    "url": "ws://localhost:8000/rpc",
    "namespace": "rapmat",
    "database": "main",
    "username": None,
    "password": None,
}

_GENERAL_DEFAULTS: dict[str, str] = {
    "mode": "local",
    "db_path": "",
}


# ------------------------------------------------------------------ #
#  Load / save
# ------------------------------------------------------------------ #


def load_db_config() -> dict | None:
    general: dict | None = None
    server: dict | None = None

    if _DB_CONFIG_FILE.is_file():
        with open(_DB_CONFIG_FILE, "rb") as f:
            raw = tomllib.load(f)
        general = raw.get("general", {})
        server = raw.get("server", {})

    if general is None and server is None:
        return None

    return {
        "general": {**_GENERAL_DEFAULTS, **(general or {})},
        "server": {**_SERVER_DEFAULTS, **(server or {})},
    }


def get_active_mode() -> str:
    full = load_db_config()
    if full is None:
        return "local"
    return full.get("general", {}).get("mode", "local")


def save_db_config(
    *,
    general: dict | None = None,
    server: dict | None = None,
) -> None:
    existing = load_db_config() or {
        "general": dict(_GENERAL_DEFAULTS),
        "server": dict(_SERVER_DEFAULTS),
    }
    gen = {**existing["general"], **(general or {})}
    srv = {**existing["server"], **(server or {})}

    APP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["[general]"]
    for key in ("mode", "db_path"):
        lines.append(f'{key} = "{gen.get(key, "")}"')
    lines.append("")
    lines.append("[server]")
    for key in ("url", "namespace", "database", "username", "password"):
        lines.append(f'{key} = "{srv.get(key, "") or ""}"')
    _DB_CONFIG_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clear_db_config() -> bool:
    if _DB_CONFIG_FILE.is_file():
        _DB_CONFIG_FILE.unlink()
        return True
    return False


# ------------------------------------------------------------------ #
#  Store resolution
# ------------------------------------------------------------------ #

_DEFAULT_SURREAL_PATH = str(APP_DATA_DIR / "surrealdb")


def resolve_store() -> StructureStore:
    full = load_db_config()
    mode = (full or {}).get("general", {}).get("mode", "local") if full else "local"
    custom_path = (full or {}).get("general", {}).get("db_path", "") if full else ""

    if mode == "remote":
        srv = (full or {}).get("server", {})
        if srv and srv.get("url"):
            return _make_surreal_remote(srv)
        return _make_surreal_local(Path(custom_path or _DEFAULT_SURREAL_PATH))

    path = custom_path or _DEFAULT_SURREAL_PATH
    return _make_surreal_local(Path(path))


# ------------------------------------------------------------------ #
#  Backend constructors
# ------------------------------------------------------------------ #


def _make_surreal_local(db_path: Path) -> "StructureStore":
    from rapmat.storage.surrealdb_store import SurrealDBStore

    return SurrealDBStore.from_path(db_path)


def _make_surreal_remote(srv: dict) -> "StructureStore":
    from rapmat.storage.surrealdb_store import SurrealDBStore

    return SurrealDBStore(
        db_url=srv["url"],
        namespace=srv.get("namespace", "rapmat"),
        database=srv.get("database", "main"),
        username=srv.get("username"),
        password=srv.get("password"),
    )
