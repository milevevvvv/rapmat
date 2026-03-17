"""Convex hull screen for the Rapmat TUI."""

from pathlib import Path


import urwid

from rapmat.tui.widgets.form import (
    FormGroup,
    checkbox_field,
    dropdown_field,
    float_field,
    text_field,
)
from rapmat.tui.widgets.table import SortableTable

from rapmat.tui.router import ScreenRouter
from rapmat.tui.state import AppState

_HULL_COLS = [
    ("Formula", 14),
    ("x", 8),
    ("E_form (eV/at)", 14),
    ("E_above_hull", 14),
    ("Stable", 8),
    ("Run", 22),
]


class HullScreen:
    """Convex hull viewer screen."""

    title = "Convex Hull"

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        self._state = state
        self._router = router
        self._frame: urwid.Frame | None = None
        self._structure_data: list[dict] | None = None
        self._study_system: str = ""

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
            self._state.status_bar.set_keys(
                [
                    ("F5", "Compute"),
                    ("p", "Save plot"),
                    ("Esc", "Back"),
                ]
            )
            self._state.status_bar.clear_message()

    def on_leave(self) -> None:
        pass

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
                float_field("symprec", "Symmetry precision", default=1e-3),
                text_field("save_plot", "Save plot to", default="hull.png"),
            ],
            label_width=22,
        )

        self._error_text = urwid.Text("")
        self._results_pile = urwid.Pile([])

        compute_btn = urwid.AttrMap(
            urwid.Button("Compute [F5]", on_press=self._on_compute),
            "menu_item",
            focus_map="btn_focus",
        )

        listbox_form = urwid.ListBox(
            urwid.SimpleListWalker(
                [
                    urwid.Text(("section", " Convex Hull"), align="left"),
                    urwid.Divider("─"),
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

        listbox_results = urwid.ListBox(urwid.SimpleListWalker([self._results_pile]))
        results_area = urwid.ScrollBar(
            listbox_results,
            trough_char=urwid.ScrollBar.Symbols.LITE_SHADE,
        )

        body = urwid.Pile(
            [
                ("weight", 2, form_area),
                ("weight", 3, results_area),
            ]
        )

        self._update_footer()
        return urwid.Frame(body=body)

    # ------------------------------------------------------------------ #
    #  Compute (synchronous -- fast operation)
    # ------------------------------------------------------------------ #

    def _on_compute(self, _btn=None) -> None:
        self._error_text.set_text("")
        self._results_pile.contents[:] = []

        vals = self._form.get_values()
        study_id = vals["study_id"]
        show_all = vals["show_all"]
        symprec = vals["symprec"]

        store = self._state.store

        study = store.get_study(study_id)
        if study is None:
            self._error_text.set_text(("form_error", f"Study '{study_id}' not found"))
            return

        self._study_system = study.get("system", "")

        try:
            from rapmat.core.hull import build_phase_diagram

            pd_obj, structure_data, use_enthalpy = build_phase_diagram(
                store,
                study_id,
                symprec=symprec,
                show_all=show_all,
            )
        except Exception as e:
            self._error_text.set_text(("form_error", f"Hull error: {e}"))
            return

        self._structure_data = structure_data

        quantity = "H" if use_enthalpy else "E"

        table = SortableTable(
            columns=_HULL_COLS,
            row_data=structure_data,
            format_row=lambda sd: [
                sd["reduced_formula"],
                f"{sd['composition_frac']:.3f}",
                f"{sd['formation_energy']:.4f}",
                f"{sd['energy_above_hull']:.4f}",
                "Yes" if sd["is_stable"] else "No",
                sd["run_name"],
            ],
        )

        n_stable = sum(1 for sd in structure_data if sd["is_stable"])
        summary = urwid.Text(
            [
                ("section", f" Hull for {self._study_system}\n"),
                ("details", f"  {len(structure_data)} structures, {n_stable} stable\n"),
                (
                    "details",
                    f"  Quantity: {quantity}_form | Calculator: {study.get('calculator', '—')}",
                ),
            ]
        )

        self._results_pile.contents[:] = [
            (summary, ("pack", None)),
            (urwid.Divider("─"), ("pack", None)),
            (table, ("weight", 1)),
        ]

    # ------------------------------------------------------------------ #
    #  Save plot
    # ------------------------------------------------------------------ #

    def _save_plot(self) -> None:
        if self._structure_data is None:
            return
        vals = self._form.get_values()
        plot_path = Path(vals["save_plot"])
        try:
            from rapmat.core.hull import plot_binary_hull

            plot_binary_hull(
                self._structure_data,
                self._study_system,
                save_path=plot_path,
                show=False,
            )
            self._error_text.set_text(("success", f"Plot saved to {plot_path}"))
        except Exception as e:
            self._error_text.set_text(("form_error", f"Plot error: {e}"))

    # ------------------------------------------------------------------ #
    #  Key handling
    # ------------------------------------------------------------------ #

    def keypress(self, size: tuple, key: str) -> str | None:
        if key == "f5":
            self._on_compute()
            return None
        if key in ("p", "P"):
            self._save_plot()
            return None
        if key == "esc":
            self._router.pop()
            return None
        return key
