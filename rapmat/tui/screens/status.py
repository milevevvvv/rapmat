"""Status screen for the Rapmat TUI."""

import urwid

from rapmat.tui.widgets.table import SortableTable

from rapmat.tui.router import ScreenRouter
from rapmat.tui.state import AppState

_CALC_COLS = [
    ("Calculator", 20),
    ("Status", 14),
    ("Description", 50),
]


def _load_calc_rows() -> list[dict]:
    from rapmat.calculators import CALCULATOR_META, Calculators, is_calculator_available

    rows = []
    for calc in Calculators:
        meta = CALCULATOR_META[calc]
        error_msg = ""
        try:
            available = is_calculator_available(calc)
        except Exception as exc:
            available = False
            error_msg = str(exc)
        desc = meta.get("description", "")
        if error_msg:
            desc = f"[error: {error_msg}]"
        rows.append(
            {
                "name": calc.value,
                "available": available,
                "description": desc,
            }
        )
    return rows


def _format_calc_row(row: dict) -> list[str]:
    status = "installed" if row["available"] else "not found"
    return [row["name"], status, row["description"]]


def _calc_attr(row: dict) -> str:
    return "success" if row["available"] else "error"


class StatusScreen:
    """Environment status screen: calculators and paths."""

    title = "Status"

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        self._state = state
        self._router = router
        self._table: SortableTable | None = None
        self._widget: urwid.Widget | None = None

    # ------------------------------------------------------------------ #
    #  Screen protocol
    # ------------------------------------------------------------------ #

    def build(self) -> urwid.Widget:
        self._widget = self._build_widget()
        self._update_footer()
        return self._widget

    def on_resume(self) -> None:
        self._update_footer()

    def _update_footer(self) -> None:
        if self._state.status_bar:
            self._state.status_bar.set_keys(
                [
                    ("Esc", "Back"),
                ]
            )
            self._state.status_bar.clear_message()

    def on_leave(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    #  Layout
    # ------------------------------------------------------------------ #

    def _build_widget(self) -> urwid.Widget:
        import platformdirs

        _APP_NAME = "rapmat-materials"
        APP_CONFIG_DIR = platformdirs.user_config_dir(_APP_NAME)
        APP_DATA_DIR = platformdirs.user_data_dir(_APP_NAME)

        # Calculator table
        rows = _load_calc_rows()
        self._table = SortableTable(
            columns=_CALC_COLS,
            row_data=rows,
            format_row=_format_calc_row,
            attr_fn=_calc_attr,
        )

        # Paths section
        paths_text = urwid.Text(
            [
                ("section", " Application Paths\n"),
                ("form_label", "  Config: "),
                ("details", str(APP_CONFIG_DIR) + "\n"),
                ("form_label", "  Data:   "),
                ("details", str(APP_DATA_DIR)),
            ]
        )

        body = urwid.Pile(
            [
                ("pack", urwid.Text(("section", " Calculators"), align="left")),
                ("pack", urwid.Divider("─")),
                ("weight", 1, self._table),
                ("pack", urwid.Divider()),
                ("pack", paths_text),
                ("pack", urwid.Divider()),
                ("pack", urwid.Text(("footer", "  r Refresh  Esc Back"), align="left")),
            ]
        )

        return urwid.Padding(body, left=1, right=1)

    # ------------------------------------------------------------------ #
    #  Input
    # ------------------------------------------------------------------ #

    def keypress(self, size: tuple, key: str) -> str | None:
        if key in ("r", "R"):
            if self._table is not None:
                self._table.set_data(_load_calc_rows())
            return None
        if key == "esc":
            self._router.pop()
            return None
        return key
