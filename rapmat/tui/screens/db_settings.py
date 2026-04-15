import urwid

from rapmat.config import APP_DATA_DIR
from rapmat.tui.widgets.dialog import ModalDialog
from rapmat.tui.widgets.dropdown import DropdownSelect

from rapmat.tui.router import ScreenRouter
from rapmat.tui.state import AppState

_MODES = ["Local SurrealDB", "Remote SurrealDB"]
_MODE_TO_KEY = {
    "Local SurrealDB": "local",
    "Remote SurrealDB": "remote",
}
_KEY_TO_MODE = {v: k for k, v in _MODE_TO_KEY.items()}

_DEFAULT_SURREAL_PATH = str(APP_DATA_DIR / "surrealdb")


class DbSettingsScreen:
    title = "DB Settings"

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        self._state = state
        self._router = router
        self._frame: urwid.Frame | None = None
        self._main_body: urwid.Widget | None = None
        self._status_text: urwid.Text | None = None

        self._mode_dropdown: DropdownSelect | None = None
        self._form_pile: urwid.Pile | None = None

        self._url_edit: urwid.Edit | None = None
        self._ns_edit: urwid.Edit | None = None
        self._db_edit: urwid.Edit | None = None
        self._user_edit: urwid.Edit | None = None
        self._pass_edit: urwid.Edit | None = None

    # ------------------------------------------------------------------ #
    #  Screen protocol
    # ------------------------------------------------------------------ #

    def build(self) -> urwid.Widget:
        self._main_body = self._build_body()
        self._frame = urwid.Frame(body=self._main_body)
        self._update_footer()
        return self._frame

    def on_resume(self) -> None:
        self._update_footer()

    def _update_footer(self) -> None:
        if self._state.status_bar:
            self._state.status_bar.set_keys(
                [
                    ("Tab", "Navigate"),
                    ("Enter", "Apply"),
                    ("Esc", "Cancel"),
                ]
            )
            self._state.status_bar.clear_message()

    def on_leave(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    #  Config helpers
    # ------------------------------------------------------------------ #

    def _load(self) -> tuple[str, dict]:
        from rapmat.db_config import load_db_config

        full = load_db_config()
        if full is None:
            return "local", {
                "url": "ws://localhost:8000/rpc",
                "namespace": "rapmat",
                "database": "main",
                "username": "",
                "password": "",
            }

        mode = full.get("general", {}).get("mode", "local")
        srv = full.get("server", {})
        return mode, {
            "url": srv.get("url") or "ws://localhost:8000/rpc",
            "namespace": srv.get("namespace") or "rapmat",
            "database": srv.get("database") or "main",
            "username": srv.get("username") or "",
            "password": srv.get("password") or "",
        }

    # ------------------------------------------------------------------ #
    #  Layout
    # ------------------------------------------------------------------ #

    def _build_body(self) -> urwid.Widget:
        from rapmat.db_config import _DB_CONFIG_FILE

        mode_key, srv = self._load()
        mode_label = _KEY_TO_MODE.get(mode_key, "Local SurrealDB")
        default_idx = _MODES.index(mode_label) if mode_label in _MODES else 0

        self._mode_dropdown = DropdownSelect(
            label="Backend",
            options=_MODES,
            default=default_idx,
        )
        urwid.connect_signal(self._mode_dropdown, "change", self._on_mode_change)

        current_info = urwid.Text(
            [
                ("section", " Active Backend\n"),
                ("form_label", "  Mode:  "),
                ("details", mode_label + "\n"),
                ("form_label", "  URL:   "),
                ("details", self._state.db_url or "(embedded)" + "\n"),
                ("form_label", "  File:  "),
                ("details", str(_DB_CONFIG_FILE)),
            ]
        )

        self._url_edit = urwid.Edit(edit_text=srv["url"])
        self._ns_edit = urwid.Edit(edit_text=srv["namespace"])
        self._db_edit = urwid.Edit(edit_text=srv["database"])
        self._user_edit = urwid.Edit(edit_text=srv["username"])
        self._pass_edit = urwid.Edit(edit_text=srv["password"], mask="\u2022")

        self._form_pile = urwid.Pile([])
        self._rebuild_form(mode_key)

        self._status_text = urwid.Text("")

        test_btn = urwid.AttrMap(
            urwid.Button("Test Connection", on_press=self._on_test),
            None,
            focus_map="btn_focus",
        )
        save_btn = urwid.AttrMap(
            urwid.Button("Save", on_press=self._on_save),
            None,
            focus_map="btn_focus",
        )
        clear_btn = urwid.AttrMap(
            urwid.Button("Clear Config", on_press=self._on_clear),
            None,
            focus_map="btn_focus",
        )
        btn_row = urwid.Columns(
            [
                ("weight", 1, test_btn),
                ("weight", 1, save_btn),
                ("weight", 1, clear_btn),
            ],
            dividechars=2,
        )

        body_pile = urwid.Pile(
            [
                ("pack", current_info),
                ("pack", urwid.Divider()),
                ("pack", urwid.Text(("section", " Backend Mode"))),
                ("pack", urwid.AttrMap(self._mode_dropdown, None, focus_map="focus")),
                ("pack", urwid.Divider()),
                ("pack", self._form_pile),
                ("pack", urwid.Divider()),
                ("pack", self._status_text),
                ("pack", urwid.Divider()),
                ("pack", btn_row),
                ("pack", urwid.Divider()),
                (
                    "pack",
                    urwid.Text(("footer", "  Tab Navigate  Esc Back"), align="left"),
                ),
            ]
        )

        listbox = urwid.ListBox(urwid.SimpleListWalker([body_pile]))
        scrollable = urwid.ScrollBar(
            listbox,
            trough_char=urwid.ScrollBar.Symbols.LITE_SHADE,
        )
        return urwid.Padding(scrollable, left=2, right=2)

    def _rebuild_form(self, mode_key: str) -> None:
        if self._form_pile is None:
            return

        def _labeled(label: str, edit: urwid.Edit) -> urwid.Widget:
            lbl = urwid.Text(("form_label", label + ":"), align="right")
            return urwid.Columns(
                [
                    (14, lbl),
                    ("weight", 1, urwid.AttrMap(edit, None, focus_map="focus")),
                ],
                dividechars=1,
            )

        def _pack(w: urwid.Widget):
            return (w, ("pack", None))

        rows = []

        if mode_key == "local":
            rows.append(
                _pack(
                    urwid.Text(
                        [
                            ("form_label", "  Data path: "),
                            ("details", _DEFAULT_SURREAL_PATH),
                        ]
                    )
                )
            )
            rows.append(
                _pack(
                    urwid.Text(
                        (
                            "details",
                            "  Embedded file-based SurrealDB. No configuration needed.",
                        )
                    )
                )
            )

        elif mode_key == "remote":
            rows.append(_pack(urwid.Text(("section", " Remote SurrealDB Server"))))
            rows.append(_pack(urwid.Divider("\u2500")))
            rows.append(_pack(_labeled("URL", self._url_edit)))
            rows.append(_pack(_labeled("Namespace", self._ns_edit)))
            rows.append(_pack(_labeled("Database", self._db_edit)))
            rows.append(_pack(_labeled("Username", self._user_edit)))
            rows.append(_pack(_labeled("Password", self._pass_edit)))

        self._form_pile.contents[:] = rows

    # ------------------------------------------------------------------ #
    #  Callbacks
    # ------------------------------------------------------------------ #

    def _on_mode_change(self, _widget, value: str) -> None:
        mode_key = _MODE_TO_KEY.get(value, "local")
        self._rebuild_form(mode_key)

    def _current_mode_key(self) -> str:
        if self._mode_dropdown is None:
            return "local"
        return _MODE_TO_KEY.get(self._mode_dropdown.value, "local")

    def _get_server_cfg(self) -> dict:
        return {
            "url": self._url_edit.get_edit_text().strip() if self._url_edit else "",
            "namespace": self._ns_edit.get_edit_text().strip() if self._ns_edit else "",
            "database": self._db_edit.get_edit_text().strip() if self._db_edit else "",
            "username": (
                self._user_edit.get_edit_text().strip() if self._user_edit else ""
            ),
            "password": self._pass_edit.get_edit_text() if self._pass_edit else "",
        }

    def _on_test(self, _btn) -> None:
        if self._status_text is None:
            return
        mode = self._current_mode_key()

        if mode == "local":
            self._status_text.set_text(("details", "  Testing local SurrealDB..."))
            try:
                from rapmat.storage.surrealdb_store import SurrealDBStore

                store = SurrealDBStore.from_path(APP_DATA_DIR / "surrealdb")
                store.close()
                self._status_text.set_text(("success", "  Local SurrealDB OK."))
            except Exception as exc:
                self._status_text.set_text(
                    ("error", f"  Local SurrealDB failed: {exc}")
                )

        elif mode == "remote":
            cfg = self._get_server_cfg()
            self._status_text.set_text(("details", "  Testing remote connection..."))
            try:
                from rapmat.storage.surrealdb_store import SurrealDBStore

                store = SurrealDBStore(
                    db_url=cfg["url"],
                    namespace=cfg.get("namespace", "rapmat"),
                    database=cfg.get("database", "main"),
                    username=cfg.get("username") or None,
                    password=cfg.get("password") or None,
                )
                store.close()
                self._status_text.set_text(("success", "  Connection successful."))
            except Exception as exc:
                self._status_text.set_text(("error", f"  Connection failed: {exc}"))

    def _on_save(self, _btn) -> None:
        if self._status_text is None:
            return
        mode = self._current_mode_key()
        try:
            from rapmat.db_config import resolve_store, save_db_config
            from rapmat.config import DbBackend, DbMode, DbParams

            server_cfg = self._get_server_cfg() if mode == "remote" else None
            save_db_config(general={"mode": mode}, server=server_cfg)

            if mode == "local":
                db = DbParams(db_mode=DbMode.FILE, db_path=_DEFAULT_SURREAL_PATH)
            elif mode == "remote":
                db = DbParams(db_mode=DbMode.SERVER)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            new_store = resolve_store()
            self._state.reconnect(new_store)
            self._status_text.set_text(("success", f"  Saved & reconnected ({mode})."))
        except Exception as exc:
            self._status_text.set_text(("error", f"  Save failed: {exc}"))

    def _on_clear(self, _btn) -> None:
        if self._frame is None or self._main_body is None:
            return
        dlg = ModalDialog.confirm(
            "Clear Configuration",
            "Delete the saved DB configuration and reset to defaults?",
            parent=self._main_body,
            on_close=self._on_clear_confirm,
        )
        self._frame.body = dlg

    def _on_clear_confirm(self, confirmed: bool) -> None:
        if self._frame is None or self._main_body is None:
            return
        self._frame.body = self._main_body
        if confirmed:
            from rapmat.db_config import clear_db_config

            removed = clear_db_config()
            if self._status_text:
                if removed:
                    self._status_text.set_text(("success", "  Configuration cleared."))
                else:
                    self._status_text.set_text(
                        ("unconv", "  No configuration to clear.")
                    )

    def keypress(self, size: tuple, key: str) -> str | None:
        if key == "esc":
            self._router.pop()
            return None
        return key
