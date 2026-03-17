"""Database configuration: loading, interactive prompting, and store factory.

The persisted configuration lives in ``db.toml`` inside the user config
directory.  The file has two sections::

    [general]
    mode = "local"          # "local" | "remote"
    db_path = ""            # optional path

    [server]
    url = "ws://localhost:8000/rpc"
    namespace = "rapmat"
    database = "main"
    username = ""
    password = ""

``[general].mode`` is read by ``resolve_store`` when ``DbMode.AUTO`` is
active (the default for TUI and most CLI commands).
"""

import os
import tomllib
from pathlib import Path

from rapmat.config import APP_CONFIG_DIR, APP_DATA_DIR, DbMode
from rapmat.config import DbParams
from rapmat.storage.base import StructureStore

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
    """Read saved config from ``db.toml``, override server keys with env vars.

    Returns a dict with ``"general"`` and ``"server"`` sub-dicts, or
    ``None`` when no config file exists and no env vars are set.
    """
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
    """Return the saved ``general.mode`` string, defaulting to ``"local"``."""
    full = load_db_config()
    if full is None:
        return "local"
    return full.get("general", {}).get("mode", "local")


def save_db_config(
    *,
    general: dict | None = None,
    server: dict | None = None,
) -> None:
    """Write config to ``db.toml``.

    Merges supplied sections with current on-disk values so callers only
    need to pass the keys they want to change.
    """
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
    """Delete saved ``db.toml``. Returns ``True`` if a file was removed."""
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




