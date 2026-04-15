import urwid

from rapmat.tui.widgets.table import SortableTable

from rapmat.tui.router import ScreenRouter
from rapmat.tui.state import AppState


def _format_run_status(counts: dict[str, int]) -> str:
    parts: list[str] = []
    relaxed = counts.get("relaxed", 0)
    generating = counts.get("generating", 0)
    generated = counts.get("generated", 0)
    discarded = counts.get("discarded", 0)
    error = counts.get("error", 0)
    if relaxed:
        parts.append(f"{relaxed} relaxed")
    if generating:
        parts.append(f"{generating} generating")
    if generated:
        parts.append(f"{generated} pending")
    if discarded:
        parts.append(f"{discarded} discarded")
    if error:
        parts.append(f"{error} error")
    return " · ".join(parts) if parts else "—"


def _format_run_row(run: dict) -> list[str]:
    config = run.get("config", {})
    formula = config.get("formula", "—")
    if isinstance(formula, dict):
        formula = "".join(f"{el}{n}" if n > 1 else el for el, n in formula.items())
    ts = run.get("timestamp", "")[:16].replace("T", " ")
    status = run.get("_status_summary", "—")
    return [run.get("name", "—"), str(run.get("domain", "—")), formula, ts, status]


_RECENT_COLS = [
    ("Name", 20),
    ("Domain", 10),
    ("Formula", 12),
    ("Created", 16),
    ("Structures", 28),
]


class HomeScreen:
    title = "Home"

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        self._state = state
        self._router = router
        self._widget: urwid.Widget | None = None

    # ------------------------------------------------------------------ #
    #  Screen protocol
    # ------------------------------------------------------------------ #

    def build(self) -> urwid.Widget:
        self._state.refresh_runs_if_needed()
        self._widget = self._build_widget()
        self._update_footer()
        return self._widget

    def on_resume(self) -> None:
        self._state.refresh_runs_if_needed()
        self._update_footer()
        if self._widget is not None:
            # Rebuild the recent-runs table in-place.
            self._refresh_recent_table()

    def on_leave(self) -> None:
        pass

    def _update_footer(self) -> None:
        if self._state.status_bar:
            self._state.status_bar.set_keys(
                [
                    ("Tab", "Navigate"),
                    ("Enter", "Open"),
                    ("q", "Quit"),
                ]
            )
            self._state.status_bar.clear_message()

    # ------------------------------------------------------------------ #
    #  Layout helpers
    # ------------------------------------------------------------------ #

    def _build_widget(self) -> urwid.Widget:
        def _btn(label: str, callback) -> urwid.Widget:
            btn = urwid.Button(label, on_press=callback)
            return urwid.AttrMap(btn, "menu_item", focus_map="menu_focus")

        def _section(label: str) -> list[urwid.Widget]:
            return [urwid.Divider(), urwid.Text(("section", f" {label}"), align="left")]

        actions = urwid.Pile(
            [
                urwid.Text(("section", " Quick Actions"), align="left"),
                *_section("CSP"),
                _btn("[N] New CSP Run", self._go_new_run),
                *_section("Studies"),
                _btn("[S] Studies", self._go_studies),
                *_section("Tools"),
                _btn("[P] Phonon", self._go_phonon),
                # _btn("[F] Defects", self._go_defects),
                # NOTE: still in development
                *_section("Settings"),
                _btn("[D] DB Settings", self._go_db_settings),
                _btn("[I] Status", self._go_status),
                urwid.Divider("─"),
                _btn("[Q] Quit", self._do_quit),
            ]
        )

        left_panel = urwid.ListBox(
            urwid.SimpleListWalker(
                [
                    actions,
                    urwid.Divider(),
                    self._build_db_info(),
                ]
            )
        )

        recent_data = self._get_recent_runs()
        self._recent_table = SortableTable(
            columns=_RECENT_COLS,
            row_data=recent_data,
            format_row=_format_run_row,
            on_focus_change=None,
        )
        urwid.connect_signal(self._recent_table, "select", self._on_run_select)

        right_panel = urwid.Pile(
            [
                (
                    "pack",
                    urwid.Text(
                        ("section", " Recent Runs (↑↓ navigate, Enter to open)")
                    ),
                ),
                ("pack", urwid.Divider("─")),
                ("weight", 1, self._recent_table),
            ]
        )

        columns = urwid.Columns(
            [
                ("weight", 1, urwid.Padding(left_panel, left=2, right=2)),
                ("weight", 3, urwid.Padding(right_panel, left=1, right=1)),
            ],
            dividechars=1,
        )

        return columns

    def _build_db_info(self) -> urwid.Widget:
        url = self._state.db_url or "—"
        hw = self._get_hw_label()
        lines = [
            ("section", " Database\n"),
            ("details", f" {url}\n\n"),
            ("section", " Hardware\n"),
            ("details", f" {hw}"),
        ]
        return urwid.Text(lines)

    def _get_hw_label(self) -> str:
        try:
            import torch

            if torch.cuda.is_available():
                name = (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.device_count() > 0
                    else "Unknown GPU"
                )
                return f"⚡  CUDA ({name})"
            return "🖥️  CPU (No GPU detected)"
        except ImportError:
            return "🖥️  (Torch not installed)"

    def _get_recent_runs(self) -> list[dict]:
        runs = sorted(
            self._state.runs_cache,
            key=lambda r: r.get("timestamp", ""),
            reverse=True,
        )[:5]
        enriched = []
        for run in runs:
            counts = self._state.store.count_by_status(run["name"])
            r = dict(run)
            r["_status_summary"] = _format_run_status(counts)
            enriched.append(r)
        return enriched

    def _refresh_recent_table(self) -> None:
        recent_data = self._get_recent_runs()
        self._recent_table.set_data(recent_data)

    # ------------------------------------------------------------------ #
    #  Callbacks
    # ------------------------------------------------------------------ #

    def _go_new_run(self, _btn: urwid.Button | None = None) -> None:
        from rapmat.tui.screens.csp_search import CSPSearchScreen

        self._router.push(CSPSearchScreen(self._state, self._router))

    def _go_studies(self, _btn: urwid.Button | None = None) -> None:
        from rapmat.tui.screens.study_list import StudyListScreen

        self._router.push(StudyListScreen(self._state, self._router))

    def _go_phonon(self, _btn: urwid.Button | None = None) -> None:
        from rapmat.tui.screens.phonon import PhononDispersionScreen

        self._router.push(PhononDispersionScreen(self._state, self._router))

    def _go_db_settings(self, _btn: urwid.Button | None = None) -> None:
        from rapmat.tui.screens.db_settings import DbSettingsScreen

        self._router.push(DbSettingsScreen(self._state, self._router))

    def _go_status(self, _btn: urwid.Button | None = None) -> None:
        from rapmat.tui.screens.status import StatusScreen

        self._router.push(StatusScreen(self._state, self._router))

    def _go_defects(self, _btn: urwid.Button | None = None) -> None:
        from rapmat.tui.screens.defect import DefectScreen

        self._router.push(DefectScreen(self._state, self._router))

    def _do_quit(self, _btn: urwid.Button | None = None) -> None:
        raise urwid.ExitMainLoop()

    def _on_run_select(self, _table, run: dict) -> None:
        self._state.active_run = run["name"]
        from rapmat.tui.screens.results import ResultsScreen

        self._router.push(ResultsScreen(self._state, self._router))

    def keypress(self, size: tuple, key: str) -> str | None:
        if key in ("n",):
            self._go_new_run()
            return None
        if key in ("s", "S"):
            self._go_studies()
            return None
        if key in ("p", "P"):
            self._go_phonon()
            return None
        if key in ("d", "D"):
            self._go_db_settings()
            return None
        if key in ("i", "I"):
            self._go_status()
            return None
        if key in ("f", "F"):
            self._go_defects()
            return None
        if key in ("q", "Q"):
            self._do_quit()
        return key
