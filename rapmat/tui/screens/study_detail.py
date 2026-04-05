"""Study detail screen for the Rapmat TUI."""

import urwid

from rapmat.tui.widgets.table import SortableTable
from rapmat.tui.router import ScreenRouter
from rapmat.tui.state import AppState

_RUN_COLS = [
    ("Run Name", 28),
    ("Formula", 14),
    ("Type", 14),
    ("Structures", 16),
    ("Status", 12),
]


def _classify_run(run: dict, study_elements: list[str]) -> str:
    config = run.get("config", {})
    formula = config.get("formula", {})
    if isinstance(formula, dict):
        run_elements = set(formula.keys())
    else:
        run_elements = set()
    if len(run_elements) == 1 and run_elements <= set(study_elements):
        return "endpoint"
    return "intermediate"


def _formula_str(run: dict) -> str:
    config = run.get("config", {})
    formula = config.get("formula", {})
    if isinstance(formula, dict):
        return "".join(f"{el}{n}" if n > 1 else el for el, n in formula.items())
    return str(formula)


class StudyDetailScreen:
    """Phase diagram study detail screen."""

    title = "Study Detail"

    @property
    def breadcrumb_title(self) -> str:
        study = self._state.active_study
        return f"{study}" if study else self.title

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        self._state = state
        self._router = router
        self._table: SortableTable | None = None
        self._placeholder: urwid.WidgetPlaceholder | None = None
        self._details_text: urwid.Text | None = None
        self._details_panel: urwid.Widget | None = None

    # ------------------------------------------------------------------ #
    #  Screen protocol
    # ------------------------------------------------------------------ #

    def build(self) -> urwid.Widget:
        self._placeholder = urwid.WidgetPlaceholder(self._build_widget())
        return self._placeholder

    def on_resume(self) -> None:
        if self._placeholder is not None:
            self._placeholder.original_widget = self._build_widget()
        self._update_footer()

    def _update_footer(self) -> None:
        if self._state.status_bar:
            self._state.status_bar.set_keys(
                [
                    ("Enter", "Open Run"),
                    ("r", "Resume Run"),
                    ("u", "Unlock Run"),
                    ("n", "New Run"),
                    ("h", "Phase Analysis"),
                    ("d", "Dedup"),
                    ("Del", "Remove"),
                    ("v", "Evaluation"),
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
        study_id = self._state.active_study or ""
        study = self._state.store.get_study(study_id) if study_id else None

        if study is None:
            return urwid.Filler(
                urwid.Text(("error", f"Study '{study_id}' not found.")), valign="top"
            )

        from rapmat.utils.common import parse_system

        elements = parse_system(study.get("system", ""))
        ts = study.get("timestamp", "")[:16].replace("T", " ")

        info_text = urwid.Text(
            [
                ("section", " Study: "),
                ("details", study_id + "\n"),
                ("form_label", "  System:     "),
                ("details", study.get("system", "—") + "\n"),
                ("form_label", "  Domain:     "),
                ("details", study.get("domain", "—") + "\n"),
                ("form_label", "  Calculator: "),
                ("details", study.get("calculator", "—") + "\n"),
                ("form_label", "  Created:    "),
                ("details", ts),
            ]
        )

        runs = self._state.store.get_study_runs(study_id)
        run_rows = []
        for run in sorted(runs, key=lambda r: r.get("timestamp", "")):
            counts = self._state.store.count_by_status(run["name"])
            relaxed = counts.get("relaxed", 0)
            generated = counts.get("generated", 0)
            total = relaxed + generated
            run_type = _classify_run(run, elements)
            
            # Status display
            st = run.get("run_status", "pending")
            wid = run.get("worker_id")
            if wid and st in ("generating", "processing"):
                st = f"active({wid[:4]})"

            d = dict(run)
            d["_formula"] = _formula_str(run)
            d["_type"] = run_type
            d["_structures"] = f"{relaxed} / {total}"
            d["_status"] = st
            run_rows.append(d)

        self._table = SortableTable(
            columns=_RUN_COLS,
            row_data=run_rows,
            format_row=lambda r: [
                r["name"],
                r["_formula"],
                r["_type"],
                r["_structures"],
                r["_status"],
            ],
            attr_fn=lambda r: "warning" if r.get("worker_id") else "body",
            on_focus_change=self._on_run_focus_change,
        )
        urwid.connect_signal(self._table, "select", self._on_run_select)

        self._details_text = urwid.Text("", align="left")
        self._details_panel = urwid.BoxAdapter(
            urwid.LineBox(
                urwid.Filler(self._details_text, valign="top"),
                title="Run Configuration",
            ),
            12,
        )

        # Endpoint completeness check
        n_elements = len(elements)
        endpoint_elements: set[str] = set()
        for run in runs:
            formula = run.get("config", {}).get("formula", {})
            if isinstance(formula, dict) and len(formula) == 1:
                endpoint_elements.update(formula.keys())
        missing = set(elements) - endpoint_elements

        if n_elements < 2:
            status_text = urwid.Text(
                ("details", "  Single-element system. Phase analysis shows energy ranking.")
            )
        elif missing:
            status_text = urwid.Text(
                ("unconv", f"  Missing pure-element runs: {', '.join(sorted(missing))}")
            )
        elif n_elements == 2:
            status_text = urwid.Text(
                ("success", "  All endpoints present. Phase analysis available.")
            )
        else:
            status_text = urwid.Text(
                ("success", "  All endpoints present. Phase analysis available (table only).")
            )

        body = urwid.Pile(
            [
                ("pack", info_text),
                ("pack", urwid.Divider("─")),
                ("pack", urwid.Text(("section", " Runs in Study"), align="left")),
                ("pack", urwid.Divider("─")),
                ("weight", 1, self._table),
                ("pack", urwid.Divider()),
                ("pack", self._details_panel),
                ("pack", urwid.Divider()),
                ("pack", status_text),
            ]
        )
        if self._table:
            self._on_run_focus_change(self._table.get_focused_row())
        self._update_footer()
        return urwid.Padding(body, left=1, right=1)

    # ------------------------------------------------------------------ #
    #  Callbacks
    # ------------------------------------------------------------------ #

    def _on_run_focus_change(self, run: dict | None) -> None:
        if self._details_text is None:
            return
            
        markup: list = []
        if run is None:
            markup.append(("details", "No run selected.\n"))
        else:
            config = run.get("config", {})
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

    def _on_run_select(self, _table, run: dict) -> None:
        self._state.active_run = run["name"]
        from rapmat.tui.screens.results import ResultsScreen

        self._router.push(ResultsScreen(self._state, self._router))

    def _on_unlock_run(self, run_name: str) -> None:
        self._state.store.release_run(run_name, "pending")
        # Force refresh
        if self._placeholder:
            self._placeholder.original_widget = self._build_widget()

    def _open_delete_modal(self, run_name: str) -> None:
        if self._placeholder is None:
            return

        from rapmat.tui.widgets.dialog import ModalDialog

        current_body = self._placeholder.original_widget

        def _on_close(confirmed: bool) -> None:
            if self._placeholder is not None:
                self._placeholder.original_widget = current_body
                if confirmed:
                    self._state.store.delete_run(run_name)
                    self._placeholder.original_widget = self._build_widget()

        dlg = ModalDialog.confirm(
            title="Delete Run",
            message=f"Are you sure you want to permanently delete run '{run_name}' and all its structures?",
            parent=current_body,
            on_close=_on_close,
        )
        self._placeholder.original_widget = dlg

    def keypress(self, size: tuple, key: str) -> str | None:
        if key == "esc":
            self._router.pop()
            return None
        if key in ("n", "N"):
            from rapmat.tui.screens.csp_search import CSPSearchScreen
            self._router.push(CSPSearchScreen(self._state, self._router))
            return None
        if key in ("r", "R"):
            if self._table is not None:
                run = self._table.get_focused_row()
                if run:
                    self._state.active_run = run["name"]
                    from rapmat.tui.screens.csp_resume import CSPResumeScreen

                    self._router.push(CSPResumeScreen(self._state, self._router))
            return None
        if key in ("u", "U"):
            if self._table is not None:
                run = self._table.get_focused_row()
                if run:
                    self._on_unlock_run(run["name"])
            return None
        if key in ("h", "H"):
            from rapmat.tui.screens.hull import PhaseAnalysisScreen

            self._router.push(PhaseAnalysisScreen(self._state, self._router))
            return None
        if key in ("d", "D"):
            if self._table is not None:
                run = self._table.get_focused_row()
                if run:
                    self._state.active_run = run["name"]
                    from rapmat.tui.screens.dedup import DedupScreen

                    self._router.push(DedupScreen(self._state, self._router))
            return None
        if key == "delete":
            if self._table is not None:
                run = self._table.get_focused_row()
                if run:
                    self._open_delete_modal(run["name"])
            return None
        if key in ("v", "V"):
            from rapmat.tui.screens.eval import EvalScreen

            self._router.push(EvalScreen(self._state, self._router))
            return None
        return key
