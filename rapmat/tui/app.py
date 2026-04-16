import sys

import urwid

# NOTE: very dirty fix for Windows, research better solution
if sys.platform == "win32":
    sys.modules.pop("urwid.display.curses", None)

from rapmat.tui.router import ScreenRouter
from rapmat.tui.state import AppState
from rapmat.tui.widgets.status_bar import StatusBar

# ------------------------------------------------------------------ #
#  Global palette
# ------------------------------------------------------------------ #

PALETTE = [
    ("header", "white", "dark blue", "bold"),
    ("footer", "black", "light gray"),
    ("body", "light gray", "default"),
    ("focus", "white", "dark green", "standout"),
    ("error", "light red", "default"),
    ("success", "light green", "default"),
    ("unconv", "yellow", "default"),
    ("details", "white", "default"),
    ("details_title", "light cyan", "default", "bold"),
    ("section", "yellow", "default", "bold"),
    ("col_header", "white", "dark gray", "bold"),
    ("menu_item", "light gray", "default"),
    ("menu_focus", "white", "dark green", "standout"),
    ("dialog", "white", "dark gray"),
    ("btn_focus", "white", "dark green", "bold"),
    ("dropdown_hl", "white", "dark green"),
    ("menu_back", "light gray", "dark magenta"),
    ("progress", "white", "default"),
    ("pg_done", "white", "dark green", "bold"),
    ("log_line", "light cyan", "default"),
    ("form_label", "light cyan", "default"),
    ("form_error", "light red", "default"),
    ("cuda_tag", "light green", "dark blue", "bold"),
    ("cpu_tag", "light gray", "dark blue"),
    ("focus_border", "light cyan", "default", "bold"),
    ("focus_title", "light cyan", "default", "bold"),
    ("dim_border", "dark gray", "default"),
    ("dim_title", "dark gray", "default", "bold"),
]


# ------------------------------------------------------------------ #
#  Application
# ------------------------------------------------------------------ #


class RapmatApp:
    def __init__(
        self,
        state: "AppState",
        startup_error: Exception | None = None,
    ) -> None:
        from rapmat.utils.console import silence

        silence()

        self._state = state
        self._startup_error = startup_error

        self._breadcrumb = urwid.Text(" Rapmat TUI", wrap="clip")
        self._hw_status = urwid.Text(self._get_hw_status(), align="right")

        header_cols = urwid.Columns(
            [
                ("weight", 1, self._breadcrumb),
                ("pack", self._hw_status),
            ]
        )
        header = urwid.AttrMap(header_cols, "header")

        self._status_bar = StatusBar()
        self._state.status_bar = self._status_bar

        self._frame = urwid.Frame(
            body=urwid.SolidFill(" "),
            header=header,
            footer=self._status_bar,
        )

        self._router = ScreenRouter(self._frame, self._breadcrumb)

        self._loop = urwid.MainLoop(
            self._frame,
            PALETTE,
            unhandled_input=self._global_input,
            pop_ups=True,
        )

        state.loop = self._loop

        from rapmat.tui.screens.home import HomeScreen

        self._router.push(HomeScreen(self._state, self._router))

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        if self._startup_error is not None:
            self._loop.set_alarm_in(0, self._show_startup_error)
        self._loop.run()

    def _get_hw_status(self) -> list[tuple[str, str]]:
        try:
            import torch

            if torch.cuda.is_available():
                return [("cuda_tag", " ⚡  CUDA ")]
            return [("cpu_tag", " 🖥️  CPU ")]
        except ImportError:
            return [("cpu_tag", " 🖥️  NO TORCH ")]

    def _show_startup_error(self, _loop, _data) -> None:
        from rapmat.tui.widgets.dialog import ModalDialog

        exc = self._startup_error
        err_type = type(exc).__name__
        err_msg = str(exc) or "(no details)"
        message = (
            f"Could not connect to the configured database.\n\n"
            f"  {err_type}: {err_msg}\n\n"
            f"The TUI is running with a temporary in-memory store.\n"
            f"Data will NOT be persisted until the connection is fixed.\n\n"
            f"Open DB Settings to reconfigure, or Continue to proceed."
        )

        saved_body = self._frame.body

        def _open_settings():
            self._frame.body = saved_body
            from rapmat.tui.screens.db_settings import DbSettingsScreen

            self._router.push(DbSettingsScreen(self._state, self._router))

        def _continue():
            self._frame.body = saved_body

        dlg = ModalDialog.error(
            title="Database Connection Error",
            message=message,
            parent=saved_body,
            actions=[
                ("DB Settings", _open_settings),
                ("Continue", _continue),
            ],
            esc_action_index=1,
        )
        self._frame.body = dlg

    # ------------------------------------------------------------------ #
    #  Input handling
    # ------------------------------------------------------------------ #

    def _global_input(self, key: str) -> None:
        if key in ("q", "Q"):
            raise urwid.ExitMainLoop()
        current = self._router.current
        if current is not None:
            result = current.keypress((), key)
            if result is None:
                return
        if key == "esc":
            self._router.pop()
