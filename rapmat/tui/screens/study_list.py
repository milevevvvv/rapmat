"""Study list screen for the Rapmat TUI."""

import urwid

from rapmat.tui.widgets.table import SortableTable
from rapmat.tui.router import ScreenRouter
from rapmat.tui.state import AppState

_STUDY_COLS = [
    ("Name", 22),
    ("System", 14),
    ("Domain", 12),
    ("Calculator", 16),
    ("Created", 16),
    ("Runs", 6),
]


def _enrich_studies(state: "AppState") -> list[dict]:
    enriched = []
    for s in state.studies_cache:
        runs = state.store.get_study_runs(s["study_id"])
        d = dict(s)
        d["_run_count"] = len(runs)
        enriched.append(d)
    return sorted(enriched, key=lambda s: s.get("timestamp", ""), reverse=True)


def _format_study_row(row: dict) -> list[str]:
    ts = row.get("timestamp", "")[:16].replace("T", " ")
    return [
        row.get("study_id", "—"),
        row.get("system", "—"),
        row.get("domain", "—"),
        row.get("calculator", "—"),
        ts,
        str(row.get("_run_count", 0)),
    ]


class _SearchEdit(urwid.Edit):
    """Search box that exits on Escape."""

    def __init__(self, on_change, on_exit):
        super().__init__(caption="Search: ")
        self._on_change = on_change
        self._on_exit = on_exit

    def keypress(self, size: tuple, key: str) -> str | None:
        if key == "esc":
            self._on_exit()
            return None
        result = super().keypress(size, key)
        self._on_change(self.get_edit_text())
        return result


class StudyListScreen:
    """Phase diagram studies list screen."""

    title = "Studies"

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        self._state = state
        self._router = router
        self._all_rows: list[dict] = []
        self._table: SortableTable | None = None
        self._search_edit: _SearchEdit | None = None
        self._footer_pile: urwid.Pile | None = None
        self._sort_col: int = 0
        self._searching: bool = False
        self._widget: urwid.Widget | None = None
        self._details_text: urwid.Text | None = None
        self._details_panel: urwid.Widget | None = None

    # ------------------------------------------------------------------ #
    #  Screen protocol
    # ------------------------------------------------------------------ #

    def build(self) -> urwid.Widget:
        self._widget = urwid.WidgetPlaceholder(urwid.SolidFill())
        self._state.refresh_studies_if_needed()
        self._widget.original_widget = self._build_widget()
        self._update_footer()
        return self._widget

    def on_resume(self) -> None:
        self._state.refresh_studies_if_needed()
        self._update_footer()
        if self._table is not None:
            self._all_rows = _enrich_studies(self._state)
            self._table.set_data(self._all_rows)

    def _update_footer(self) -> None:
        if self._state.status_bar:
            self._state.status_bar.set_keys(
                [
                    (" Enter", "Open"),
                    (" n", "New"),
                    (" /", "Search"),
                    (" s", "Sort"),
                    (" Del", "Remove"),
                    (" Esc", "Back"),
                ]
            )
            self._state.status_bar.clear_message()

    def on_leave(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    #  Layout
    # ------------------------------------------------------------------ #

    def _build_widget(self) -> urwid.Widget:
        self._all_rows = _enrich_studies(self._state)

        self._table = SortableTable(
            columns=_STUDY_COLS,
            row_data=self._all_rows,
            format_row=_format_study_row,
            on_focus_change=self._on_study_focus_change,
        )
        urwid.connect_signal(self._table, "select", self._on_study_select)

        self._details_text = urwid.Text("", align="left")
        self._details_panel = urwid.BoxAdapter(
            urwid.LineBox(
                urwid.Filler(self._details_text, valign="top"),
                title="Study Configuration",
            ),
            12,
        )

        self._search_edit = _SearchEdit(
            on_change=self._apply_search,
            on_exit=self._exit_search,
        )

        self._footer_pile = urwid.Pile([])

        body = urwid.Pile(
            [
                (
                    "pack",
                    urwid.Text(("section", " Phase Diagram Studies"), align="left"),
                ),
                ("pack", urwid.Divider("─")),
                ("weight", 1, self._table),
                ("pack", urwid.Divider()),
                ("pack", self._details_panel),
                ("pack", urwid.Divider()),
                ("pack", self._footer_pile),
            ]
        )
        if self._table:
            self._on_study_focus_change(self._table.get_focused_row())
        return urwid.Padding(body, left=1, right=1)

    # ------------------------------------------------------------------ #
    #  Search helpers
    # ------------------------------------------------------------------ #

    def _apply_search(self, query: str) -> None:
        if not query:
            self._table.set_data(self._all_rows)
            return
        q = query.lower()
        filtered = [
            r
            for r in self._all_rows
            if q in r.get("study_id", "").lower()
            or q in r.get("system", "").lower()
            or q in r.get("domain", "").lower()
            or q in r.get("calculator", "").lower()
        ]
        self._table.set_data(filtered)

    def _enter_search(self) -> None:
        self._searching = True
        if self._search_edit and self._footer_pile:
            self._search_edit.set_edit_text("")
            self._footer_pile.contents = [
                (self._search_edit, self._footer_pile.options()),
            ]
            self._footer_pile.focus_position = 0

    def _exit_search(self) -> None:
        self._searching = False
        if self._footer_pile:
            self._footer_pile.contents = []
        self._apply_search("")

    # ------------------------------------------------------------------ #
    #  Callbacks
    # ------------------------------------------------------------------ #

    def _on_study_focus_change(self, study: dict | None) -> None:
        if self._details_text is None:
            return
        
        markup: list = []
        if study is None:
            markup.append(("details", "No study selected.\n"))
        else:
            config = study.get("config", {})
            if not config:
                markup.append(("details", "No configuration parameters available.\n"))
            else:
                for k in sorted(config.keys()):
                    val = config[k]
                    if isinstance(val, dict):
                        import json
                        try:
                            val_str = json.dumps(val)
                        except Exception:
                            val_str = str(val)
                    else:
                        val_str = str(val)
                    label = str(k).replace("_", " ").title() + ":"
                    markup.append(("form_label", f"  {label:<22} "))
                    markup.append(("details", f"{val_str}\n"))
        self._details_text.set_text(markup)

    def _on_study_select(self, _table, study: dict) -> None:
        self._state.active_study = study["study_id"]
        from rapmat.tui.screens.study_detail import StudyDetailScreen

        self._router.push(StudyDetailScreen(self._state, self._router))

    def keypress(self, size: tuple, key: str) -> str | None:
        if self._searching:
            if self._search_edit:
                return self._search_edit.keypress(size, key)
            return key
        if key == "/":
            self._enter_search()
            return None
        if key in ("n", "N"):
            from rapmat.tui.screens.study_create import StudyCreateScreen

            self._router.push(StudyCreateScreen(self._state, self._router))
            return None
        if key in ("s", "S"):
            self._sort_col = (self._sort_col + 1) % len(_STUDY_COLS)
            if self._table:
                self._table.sort_by(self._sort_col)
            return None
        if key == "delete":
            if self._table is not None:
                study = self._table.get_focused_row()
                if study:
                    self._open_delete_modal(study["study_id"])
            return None
        return key

    def _open_delete_modal(self, study_id: str) -> None:
        if self._widget is None:
            return

        from rapmat.tui.widgets.dialog import ModalDialog

        current_body = self._widget.original_widget

        def _on_close(confirmed: bool) -> None:
            if self._widget is not None:
                self._widget.original_widget = current_body
                if confirmed:
                    self._state.store.delete_study(study_id)
                    self._state.invalidate_studies()
                    self._state.refresh_studies_if_needed()
                    self._all_rows = _enrich_studies(self._state)
                    if self._table is not None:
                        self._table.set_data(self._all_rows)
                        self._on_study_focus_change(self._table.get_focused_row())

        dlg = ModalDialog.confirm(
            title="Delete Study",
            message=(
                f"Are you sure you want to permanently delete study '{study_id}'?\n\n"
                "This will recursively remove:\n"
                " - All runs belonging to this study\n"
                " - All structures belonging to those runs"
            ),
            parent=current_body,
            on_close=_on_close,
        )
        self._widget.original_widget = dlg
