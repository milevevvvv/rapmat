"""Phase analysis screen for the Rapmat TUI."""

from pathlib import Path


import urwid

from rapmat.tui.widgets.dialog import ModalDialog
from rapmat.tui.widgets.form import (
    FormGroup,
    checkbox_field,
    dropdown_field,
    float_field,
)
from rapmat.tui.widgets.table import SortableTable

from rapmat.tui.router import ScreenRouter
from rapmat.tui.state import AppState

# ------------------------------------------------------------------ #
#  Column definitions per system type
# ------------------------------------------------------------------ #

_UNARY_COLS = [
    ("Formula", 14),
    ("E/atom (eV)", 14),
    ("Run", 28),
]

_BINARY_COLS = [
    ("Formula", 14),
    ("x", 8),
    ("E/atom (eV)", 14),
    ("E_form (eV/at)", 14),
    ("EAH (eV/at)", 14),
    ("Stable", 8),
    ("Run", 22),
]

_MULTI_COLS = [
    ("Formula", 14),
    ("E/atom (eV)", 14),
    ("E_form (eV/at)", 14),
    ("EAH (eV/at)", 14),
    ("Stable", 8),
    ("Run", 22),
]


class PhaseAnalysisScreen:
    """Phase analysis viewer — supports unary, binary, and ternary+ systems."""

    title = "Phase Analysis"

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        self._state = state
        self._router = router
        self._frame: urwid.Frame | None = None
        self._main_body: urwid.Widget | None = None
        self._structure_data: list[dict] | None = None
        self._study_system: str = ""
        self._system_size: int = 0  # 1=unary, 2=binary, 3+=ternary+
        self._use_enthalpy: bool = False

    # ------------------------------------------------------------------ #
    #  Screen protocol
    # ------------------------------------------------------------------ #

    def build(self) -> urwid.Widget:
        self._state.refresh_studies_if_needed()
        self._frame = self._build_frame()
        return self._frame

    def on_resume(self) -> None:
        self._update_footer()

    def on_leave(self) -> None:
        pass

    def _update_footer(self) -> None:
        if self._state.status_bar:
            keys = [
                ("F5", "Compute"),
                ("Esc", "Back"),
            ]
            if self._system_size == 2 and self._structure_data is not None:
                keys.insert(1, ("p", "Save Plot"))
            self._state.status_bar.set_keys(keys)
            self._state.status_bar.clear_message()

    # ------------------------------------------------------------------ #
    #  Layout
    # ------------------------------------------------------------------ #

    def _study_options(self) -> list[str]:
        names = []
        for s in self._state.studies_cache:
            sid = s.get("study_id") or s.get("id") or s.get("name", "")
            if isinstance(sid, str) and ":" in sid:
                sid = sid.split(":")[-1]
            names.append(str(sid))
        return names if names else ["(no studies)"]

    def _build_frame(self) -> urwid.Frame:
        study_opts = self._study_options()
        default_idx = 0
        if self._state.active_study:
            sid = self._state.active_study
            if isinstance(sid, str) and ":" in sid:
                sid = sid.split(":")[-1]
            if sid in study_opts:
                default_idx = study_opts.index(sid)

        self._form = FormGroup(
            [
                dropdown_field("study_id", "Study", study_opts, default=default_idx),
                checkbox_field("show_all", "Show all structures", default=False),
            ],
            label_width=22,
        )

        self._error_text = urwid.Text("")
        self._results_placeholder = urwid.WidgetPlaceholder(urwid.SolidFill(" "))

        compute_btn = urwid.AttrMap(
            urwid.Button("Compute [F5]", on_press=self._on_compute),
            "menu_item",
            focus_map="btn_focus",
        )

        listbox_form = urwid.ListBox(
            urwid.SimpleListWalker(
                [
                    self._form,
                    urwid.Divider(),
                    urwid.Columns([(18, compute_btn)], dividechars=1),
                    self._error_text,
                ]
            )
        )
        form_area = urwid.ScrollBar(
            listbox_form,
            trough_char=urwid.ScrollBar.Symbols.LITE_SHADE,
        )

        body = urwid.Pile(
            [
                ("weight", 2, form_area),
                ("weight", 3, self._results_placeholder),
            ]
        )

        self._main_body = body
        self._update_footer()
        return urwid.Frame(body=body)

    # ------------------------------------------------------------------ #
    #  Compute
    # ------------------------------------------------------------------ #

    def _on_compute(self, _btn=None) -> None:
        self._error_text.set_text("")
        self._results_placeholder.original_widget = urwid.SolidFill(" ")
        self._structure_data = None

        vals = self._form.get_values()
        study_id = vals["study_id"]
        show_all = vals["show_all"]

        store = self._state.store

        study = store.get_study(study_id)
        if study is None:
            self._error_text.set_text(("form_error", f"Study '{study_id}' not found"))
            return

        symprec = study.get("config", {}).get("symprec", 1e-3)
        self._study_system = study.get("system", "")

        from rapmat.utils.common import parse_system
        elements = parse_system(self._study_system)
        self._system_size = len(elements)

        try:
            if self._system_size < 2:
                # Unary: energy ranking only
                from rapmat.core.hull import build_energy_ranking
                structure_data = build_energy_ranking(store, study_id)
                self._use_enthalpy = False
                self._structure_data = structure_data
                self._show_unary_results(structure_data)
            else:
                # Binary or ternary+: full phase diagram
                from rapmat.core.hull import build_phase_diagram
                pd_obj, structure_data, use_enthalpy = build_phase_diagram(
                    store, study_id, symprec=symprec, show_all=show_all,
                )
                self._use_enthalpy = use_enthalpy
                self._structure_data = structure_data
                if self._system_size == 2:
                    self._show_binary_results(structure_data, study, use_enthalpy)
                else:
                    self._show_multi_results(structure_data, study, use_enthalpy)
        except Exception as e:
            self._error_text.set_text(("form_error", f"Error: {e}"))
            return

        self._update_footer()

    # ------------------------------------------------------------------ #
    #  Result display per system type
    # ------------------------------------------------------------------ #

    def _show_unary_results(self, data: list[dict]) -> None:
        table = SortableTable(
            columns=_UNARY_COLS,
            row_data=data,
            format_row=lambda sd: [
                sd["reduced_formula"],
                f"{sd['energy_per_atom']:.4f}",
                sd["run_name"],
            ],
        )

        summary = urwid.Text(
            [
                ("section", f" Energy Ranking — {self._study_system}\n"),
                ("details", f"  {len(data)} structures, sorted by E/atom\n"),
            ]
        )

        self._results_placeholder.original_widget = urwid.Pile(
            [
                ("pack", summary),
                ("pack", urwid.Divider("─")),
                ("weight", 1, table),
            ]
        )

    def _show_binary_results(
        self, data: list[dict], study: dict, use_enthalpy: bool
    ) -> None:
        quantity = "H" if use_enthalpy else "E"

        table = SortableTable(
            columns=_BINARY_COLS,
            row_data=data,
            format_row=lambda sd: [
                sd["reduced_formula"],
                f"{sd['composition_frac']:.3f}",
                f"{sd['effective_per_atom']:.4f}",
                f"{sd['formation_energy']:.4f}",
                f"{sd['energy_above_hull']:.4f}",
                "Yes" if sd["is_stable"] else "No",
                sd["run_name"],
            ],
        )

        n_stable = sum(1 for sd in data if sd["is_stable"])
        symprec = study.get("config", {}).get("symprec", 1e-3)
        summary = urwid.Text(
            [
                ("section", f" Phase Analysis — {self._study_system}\n"),
                ("details", f"  {len(data)} structures, {n_stable} stable\n"),
                (
                    "details",
                    f"  Quantity: {quantity}_form | Calculator: {study.get('calculator', '—')} | Symprec: {symprec}",
                ),
            ]
        )

        self._results_placeholder.original_widget = urwid.Pile(
            [
                ("pack", summary),
                ("pack", urwid.Divider("─")),
                ("weight", 1, table),
            ]
        )

    def _show_multi_results(
        self, data: list[dict], study: dict, use_enthalpy: bool
    ) -> None:
        quantity = "H" if use_enthalpy else "E"

        table = SortableTable(
            columns=_MULTI_COLS,
            row_data=data,
            format_row=lambda sd: [
                sd["reduced_formula"],
                f"{sd['effective_per_atom']:.4f}",
                f"{sd['formation_energy']:.4f}",
                f"{sd['energy_above_hull']:.4f}",
                "Yes" if sd["is_stable"] else "No",
                sd["run_name"],
            ],
        )

        n_stable = sum(1 for sd in data if sd["is_stable"])
        symprec = study.get("config", {}).get("symprec", 1e-3)
        summary = urwid.Text(
            [
                ("section", f" Phase Analysis — {self._study_system}\n"),
                ("details", f"  {len(data)} structures, {n_stable} stable\n"),
                (
                    "details",
                    f"  Quantity: {quantity}_form | Calculator: {study.get('calculator', '—')} | Symprec: {symprec}\n",
                ),
                ("unconv", "  Plot not available for ternary+ systems."),
            ]
        )

        self._results_placeholder.original_widget = urwid.Pile(
            [
                ("pack", summary),
                ("pack", urwid.Divider("─")),
                ("weight", 1, table),
            ]
        )

    # ------------------------------------------------------------------ #
    #  Save plot (binary only, via modal)
    # ------------------------------------------------------------------ #

    def _open_save_plot_modal(self) -> None:
        if self._structure_data is None or self._system_size != 2:
            return
        if self._frame is None or self._main_body is None:
            return

        dlg = ModalDialog.input_text(
            title="Save Hull Plot",
            message="Enter file path for the hull plot:",
            parent=self._main_body,
            on_save=self._do_save_plot,
            on_cancel=self._close_modal,
            default="hull.png",
        )
        self._frame.body = dlg

    def _do_save_plot(self, path_str: str) -> None:
        self._close_modal()
        plot_path = Path(path_str)
        try:
            from rapmat.core.hull import plot_binary_hull

            plot_binary_hull(
                self._structure_data,
                self._study_system,
                save_path=plot_path,
                show=False,
                use_enthalpy=self._use_enthalpy,
            )
            self._error_text.set_text(("success", f"Plot saved to {plot_path}"))
        except Exception as e:
            self._error_text.set_text(("form_error", f"Plot error: {e}"))

    def _close_modal(self) -> None:
        if self._frame and self._main_body:
            self._frame.body = self._main_body

    # ------------------------------------------------------------------ #
    #  Key handling
    # ------------------------------------------------------------------ #

    def keypress(self, size: tuple, key: str) -> str | None:
        if key == "f5":
            self._on_compute()
            return None
        if key in ("p", "P"):
            self._open_save_plot_modal()
            return None
        if key == "esc":
            self._router.pop()
            return None
        return key
