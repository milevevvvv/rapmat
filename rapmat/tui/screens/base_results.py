import math
import urwid

from ase import Atoms
from ase.io import write as write_ase_structure
from pathlib import Path
from typing import Optional

from rapmat.tui.widgets.table import SortableTable
from rapmat.tui.router import ScreenRouter
from rapmat.tui.state import AppState
from rapmat.tui.tasks import BackgroundTask


def _dyn_stability(result: dict, phonon_cutoff: float | None) -> Optional[bool]:
    min_freq = result.get("min_phonon_freq")
    if min_freq is not None and phonon_cutoff is not None:
        try:
            f = float(min_freq)
            if not math.isnan(f):
                return f >= phonon_cutoff
        except (TypeError, ValueError):
            pass
    return result.get("dynamical_stability")


def _row_attr(result: dict, phonon_cutoff: float | None) -> str:
    if not result.get("converged", True):
        return "unconv"
    if result.get("duplicate") is True:
        return "unconv"
    dyn = _dyn_stability(result, phonon_cutoff)
    if dyn is False:
        return "error"
    return "body"


class _SearchEdit(urwid.Edit):
    def __init__(self, on_search, on_exit) -> None:
        super().__init__(caption="Search: ")
        self._on_search = on_search
        self._on_exit = on_exit

    def keypress(self, size: tuple, key: str) -> str | None:
        if key == "enter":
            self._on_search(self.edit_text)
            self._on_exit()
            return None
        if key == "esc":
            self._on_exit()
            return None
        return super().keypress(size, key)


class _ResultsFooter(urwid.WidgetWrap):
    def __init__(self, screen: "BaseResultsScreen") -> None:
        self._screen = screen
        self._search = _SearchEdit(screen.apply_search, screen.exit_search)
        self._pile = urwid.Pile([])
        super().__init__(self._pile)

    def show_search(self) -> None:
        self._search.set_edit_text("")
        self._pile.contents = [(self._search, self._pile.options())]

    def show_status(self, message: str = "") -> None:
        self._pile.contents = []
        self._screen.update_footer_keys(message)


class _SaveDialog(urwid.WidgetWrap):
    signals = ["close"]

    def __init__(
        self,
        parent: urwid.Widget,
        on_save,
        num_filtered: int = 1,
        default_dir: str = "",
    ) -> None:
        self._on_save = on_save

        self._scope_group: list = []
        scope_rb1 = urwid.RadioButton(self._scope_group, "Focused structure only")
        self._save_all_rb = urwid.RadioButton(
            self._scope_group, f"All {num_filtered} filtered structures"
        )

        self._fmt_group: list = []
        cif_rb = urwid.RadioButton(self._fmt_group, "cif")
        xyz_rb = urwid.RadioButton(self._fmt_group, "xyz")
        if not default_dir:
            default_dir = str(Path.cwd())
        self._dir_edit = urwid.Edit(caption="Directory: ", edit_text=default_dir)
        self._standardize_cb = urwid.CheckBox("Standardize cell", state=False)

        def _ok(_btn: urwid.Button) -> None:
            save_all = self._save_all_rb.state
            fmt = next((rb.label for rb in self._fmt_group if rb.state), "cif")
            directory = self._dir_edit.edit_text.strip() or default_dir
            standardize = self._standardize_cb.state
            self._emit("close", True)
            on_save(fmt, directory, standardize, save_all)

        def _cancel(_btn: urwid.Button) -> None:
            self._emit("close", False)

        ok_btn = urwid.AttrMap(
            urwid.Button("Save", on_press=_ok), None, focus_map="btn_focus"
        )
        cancel_btn = urwid.AttrMap(
            urwid.Button("Cancel", on_press=_cancel), None, focus_map="btn_focus"
        )

        body = urwid.Pile(
            [
                urwid.Text("Scope:"),
                scope_rb1,
                self._save_all_rb,
                urwid.Divider(),
                urwid.Text("Format:"),
                cif_rb,
                xyz_rb,
                urwid.Divider(),
                self._dir_edit,
                urwid.Divider(),
                self._standardize_cb,
                urwid.Divider(),
                urwid.Columns(
                    [("weight", 1, ok_btn), ("weight", 1, cancel_btn)],
                    dividechars=2,
                ),
            ]
        )
        inner = urwid.LineBox(
            urwid.Padding(body, left=1, right=1), title="Save Structure"
        )
        overlay = urwid.Overlay(
            urwid.AttrMap(inner, "dialog"),
            parent,
            align=urwid.CENTER,
            width=(urwid.RELATIVE, 50),
            valign=urwid.MIDDLE,
            height=urwid.PACK,
            min_width=45,
        )
        super().__init__(overlay)

    def keypress(self, size: tuple, key: str) -> str | None:
        if key == "esc":
            self._emit("close", False)
            return None
        return super().keypress(size, key)


class BaseResultsScreen:
    title = "Base Results"

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        self._state = state
        self._router = router

        self._pressure_gpa: float = 0.0
        self._thickness_cutoff: float | None = None
        self._phonon_cutoff: float | None = None

        self._results: list[dict] = []
        self._structures: list[Atoms] = []

        self._hide_unconverged: bool = True
        self._hide_thick: bool = False
        self._hide_duplicates: bool = False
        self._search_query: str = ""

        self._show_thickness: bool = False
        self._show_dynamical_stability: bool = False
        self._show_duplicate_col: bool = False

        self._app_message: str = ""

        self._main_frame: urwid.Frame | None = None
        self._table: SortableTable | None = None
        self._details_content: urwid.WidgetPlaceholder | None = None
        self._footer: _ResultsFooter | None = None
        self._details_panel: urwid.Widget | None = None
        self._body_pile: urwid.Pile | None = None
        self._phonon_task: "BackgroundTask | None" = None

    def build(self) -> urwid.Widget:
        self._fetch_data()
        self._main_frame = self._build_frame()
        return self._main_frame

    def on_resume(self) -> None:
        self.update_footer_keys()

    def on_leave(self) -> None:
        pass

    def update_footer_keys(self, message: str = "") -> None:
        pass

    def _fetch_data(self) -> None:
        pass

    def _columns_def(self) -> list[tuple[str, int]]:
        return []

    def _format_row(self, result: dict) -> list[str]:
        return []

    def _get_symprec(self) -> float:
        return 1e-3

    def _get_extra_details(self, result: dict) -> list:
        return []

    def _on_phonon_complete(self, phonon_cutoff: float) -> None:
        pass

    def _attr_fn(self, result: dict) -> str:
        return _row_attr(result, self._phonon_cutoff)

    def _build_frame(self) -> urwid.Frame:
        cols = self._columns_def()
        display = self._get_display_results()

        self._table = SortableTable(
            columns=cols,
            row_data=display,
            format_row=self._format_row,
            attr_fn=self._attr_fn,
            on_focus_change=self._on_focus_change,
        )
        urwid.connect_signal(self._table, "select", self._on_row_select)

        self._details_content = urwid.WidgetPlaceholder(
            urwid.Text("No structure selected.")
        )
        self._details_panel = urwid.LineBox(
            self._details_content,
            title="Structure Details",
        )

        self._footer = _ResultsFooter(self)

        self._body_pile = urwid.Pile(
            [
                ("weight", 1, urwid.AttrMap(self._table, "body")),
                ("pack", self._details_panel),
            ]
        )

        frame = urwid.Frame(
            body=self._body_pile,
            footer=self._footer,
        )

        self._update_details(self._table.get_focused_row())
        self.update_footer_keys()

        return frame

    def _get_display_results(self) -> list[dict]:
        res = self._results
        if self._hide_unconverged:
            res = [r for r in res if r.get("converged", True)]
        if self._hide_thick and self._thickness_cutoff is not None:
            res = [
                r
                for r in res
                if r.get("thickness") is not None
                and r["thickness"] <= self._thickness_cutoff
            ]
        if self._hide_duplicates:
            res = [r for r in res if r.get("duplicate") is not True]
        if self._search_query:
            q = self._search_query
            res = [
                r
                for r in res
                if q in " ".join(str(v).lower() for v in r.values() if v is not None)
            ]
        return res

    def _rebuild_table(self) -> None:
        if self._table is None:
            return
        new_cols = self._columns_def()
        self._table.update_columns(new_cols)
        self._table.set_data(self._get_display_results())

    def _update_details(self, result: dict | None) -> None:
        if getattr(self, "_details_content", None) is None:
            return

        if result is None:
            if (
                self._hide_unconverged
                and self._results
                and not any(r.get("converged", True) for r in self._results)
            ):
                self._details_content.original_widget = urwid.Text(
                    [
                        (
                            "details",
                            "All structures are unconverged. Press [u] to show them.",
                        )
                    ]
                )
            else:
                self._details_content.original_widget = urwid.Text(
                    [("details", "No structure selected.")]
                )
            return

        idx = result.get("structure_index", result.get("id", 1) - 1)
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            idx = None

        if idx is not None and 0 <= idx < len(self._structures):
            atoms = self._structures[idx]
            cell_lengths = atoms.get_cell().lengths()

            cells = []

            def add_cell(label, val):
                cells.append(
                    urwid.Text(
                        [("form_label", f"{label}: "), ("details", str(val))],
                        align="left",
                    )
                )

            if result.get("structure_id"):
                add_cell("ID", result.get("structure_id"))

            add_cell("Atoms", len(atoms))
            add_cell(
                "Cell (Å)",
                f"{cell_lengths[0]:.3f}, {cell_lengths[1]:.3f}, {cell_lengths[2]:.3f}",
            )
            add_cell("Initial SG", result.get("initial_spg", "N/A"))
            add_cell("Final SG", result.get("final_spg", "N/A"))

            epa = result.get("energy_per_atom", result.get("effective_per_atom", 0.0))
            add_cell("Energy/atom", f"{epa:.4f} eV")

            if self._pressure_gpa > 0:
                h = result.get("enthalpy_per_atom")
                if h is not None:
                    add_cell("Enthalpy/atom", f"{h:.4f} eV")
                vol = result.get("volume")
                if vol is not None:
                    add_cell("Volume", f"{vol:.3f} Å³")
                add_cell("Pressure", f"{self._pressure_gpa} GPa")

            fmax = result.get("fmax")
            if fmax is not None:
                add_cell("Fmax", f"{fmax:.3f}")

            converged = result.get("converged")
            if converged is not None:
                add_cell("Converged", bool(converged))

            if self._show_thickness:
                t = result.get("thickness")
                if t is not None:
                    add_cell("Thickness (Å)", f"{t:.2f}")

            for extra in self._get_extra_details(result):
                markup, text = extra
                text = text.rstrip("\n")
                if ":" in text:
                    label, val = text.split(":", 1)
                    cells.append(
                        urwid.Text(
                            [("form_label", f"{label}: "), ("details", val.strip())],
                            align="left",
                        )
                    )
                else:
                    cells.append(urwid.Text([(markup, text)], align="left"))

            if self._show_dynamical_stability:
                dyn = _dyn_stability(result, self._phonon_cutoff)
                dyn_str = "Yes" if dyn is True else ("No" if dyn is False else "N/A")
                add_cell("Dyn. Stability", dyn_str)

            min_freq = result.get("min_phonon_freq")
            if min_freq is not None:
                try:
                    f = float(min_freq)
                    if not math.isnan(f):
                        add_cell("Min freq (THz)", f"{f:.4f}")
                except (TypeError, ValueError):
                    pass

            dup = result.get("duplicate")
            if dup is not None:
                add_cell("Duplicate", "Yes" if dup else "No")

            grid = urwid.GridFlow(cells, cell_width=35, h_sep=2, v_sep=1, align="left")
            self._details_content.original_widget = grid
        else:
            self._details_content.original_widget = urwid.Text(
                [("details", "No structure data available for this row.")]
            )

    def apply_search(self, query: str) -> None:
        self._search_query = query.strip().lower()
        self._rebuild_table()
        if self._search_query:
            self._show_message(f"Filtered: {self._search_query!r}")
        else:
            self._show_message("")

    def exit_search(self) -> None:
        if self._main_frame is None or self._footer is None:
            return
        self._footer.show_status(self._app_message)
        self._main_frame.focus_position = "body"

    def _show_message(self, msg: str) -> None:
        self._app_message = msg
        if self._footer:
            self._footer.show_status(msg)

    def _action_toggle_unconverged(self) -> None:
        self._hide_unconverged = not self._hide_unconverged
        self._rebuild_table()
        self._show_message(
            "Hiding unconverged." if self._hide_unconverged else "Showing unconverged."
        )

    def _action_toggle_duplicates(self) -> None:
        if not self._show_duplicate_col:
            self._show_message("No dedup data. Run dedup analysis and apply first.")
            return
        self._hide_duplicates = not self._hide_duplicates
        self._rebuild_table()
        self._show_message(
            "Hiding duplicates." if self._hide_duplicates else "Showing duplicates."
        )

    def _action_thickness(self) -> None:
        if self._main_frame is None:
            return
        if not self._show_thickness:
            self._show_message("No thickness data available for this run.")
            return

        from rapmat.tui.widgets.form import FormGroup, float_field

        default_val = (
            self._thickness_cutoff if self._thickness_cutoff is not None else 0.0
        )
        form = FormGroup(
            fields=[
                float_field("cutoff", "Max thickness (Å)", default=default_val),
            ],
            label_width=18,
        )

        current_body = self._main_frame.body

        def _on_apply(_btn) -> None:
            errors = form.validate()
            if errors:
                err_text.set_text(("form_error", "; ".join(errors)))
                return
            vals = form.get_values()
            cutoff = float(vals.get("cutoff", 0.0))
            self._main_frame.body = current_body
            if cutoff > 0:
                self._thickness_cutoff = cutoff
                self._hide_thick = True
                self._rebuild_table()
                self._show_message(f"Hiding thickness > {cutoff:.2f} Å.")
            else:
                self._thickness_cutoff = None
                self._hide_thick = False
                self._rebuild_table()
                self._show_message("Showing all thicknesses.")

        def _on_cancel(_btn) -> None:
            self._main_frame.body = current_body

        def _on_clear(_btn) -> None:
            self._main_frame.body = current_body
            self._thickness_cutoff = None
            self._hide_thick = False
            self._rebuild_table()
            self._show_message("Showing all thicknesses.")

        err_text = urwid.Text("")
        apply_btn = urwid.AttrMap(
            urwid.Button("Apply", on_press=_on_apply),
            None,
            focus_map="btn_focus",
        )
        clear_btn = urwid.AttrMap(
            urwid.Button("Clear", on_press=_on_clear),
            None,
            focus_map="btn_focus",
        )
        cancel_btn = urwid.AttrMap(
            urwid.Button("Cancel", on_press=_on_cancel),
            None,
            focus_map="btn_focus",
        )
        btn_row = urwid.Columns(
            [
                ("weight", 1, apply_btn),
                ("weight", 1, clear_btn),
                ("weight", 1, cancel_btn),
            ],
            dividechars=2,
        )

        dialog_body = urwid.Pile(
            [
                ("pack", urwid.Text(("section", " Thickness Filter"), align="left")),
                ("pack", urwid.Divider("─")),
                ("pack", form),
                ("pack", urwid.Divider()),
                ("pack", err_text),
                ("pack", urwid.Divider()),
                ("pack", btn_row),
            ]
        )

        inner = urwid.LineBox(
            urwid.Padding(dialog_body, left=1, right=1),
            title="Thickness Filter",
        )
        overlay = urwid.Overlay(
            urwid.AttrMap(inner, "dialog"),
            current_body,
            align=urwid.CENTER,
            width=(urwid.RELATIVE, 50),
            valign=urwid.MIDDLE,
            height=urwid.PACK,
            min_width=40,
        )
        self._main_frame.body = overlay

    def _action_save(self) -> None:
        if self._table is None or self._main_frame is None:
            return
        result = self._table.get_focused_row()
        if result is None:
            self._show_message("No structure selected.")
            return
        idx = result.get("structure_index", result.get("id", 1) - 1)
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            idx = None
        if idx is None or idx < 0 or idx >= len(self._structures):
            self._show_message("No structure data available to save.")
            return

        filtered_results = self._get_display_results()
        num_filtered = len(filtered_results)

        current_body = self._main_frame.body

        def _on_save(
            fmt: str, directory: str, standardize: bool, save_all: bool
        ) -> None:
            self._main_frame.body = current_body
            if not save_all:
                self._do_save(result, idx, fmt, directory, standardize, quiet=False)
            else:
                success_count = 0
                for res in filtered_results:
                    res_idx = res.get("structure_index", res.get("id", 1) - 1)
                    try:
                        res_idx = int(res_idx)
                        if res_idx is not None and 0 <= res_idx < len(self._structures):
                            if self._do_save(
                                res, res_idx, fmt, directory, standardize, quiet=True
                            ):
                                success_count += 1
                    except (TypeError, ValueError):
                        pass
                self._show_message(
                    f"Saved {success_count}/{num_filtered} structures to {directory}"
                )

        def _on_cancel() -> None:
            self._main_frame.body = current_body

        run_name = getattr(self, "_run_name", None) or self._state.active_run
        default_dir = (
            str(Path.cwd() / f"saved_{run_name}") if run_name else str(Path.cwd())
        )

        save_dlg = _SaveDialog(
            current_body, _on_save, num_filtered=num_filtered, default_dir=default_dir
        )
        urwid.connect_signal(
            save_dlg, "close", lambda _w, ok: _on_cancel() if not ok else None
        )
        self._main_frame.body = save_dlg

    def _do_save(
        self,
        result: dict,
        idx: int,
        fmt: str,
        directory: str,
        standardize: bool = True,
        quiet: bool = False,
    ) -> bool:
        from rapmat.utils.structure import standardize_atoms

        atoms = self._structures[idx]
        if standardize:
            atoms = standardize_atoms(atoms)
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        ident = str(result.get("id", idx + 1)).replace("/", "_")
        if result.get("run_name"):
            ident = f"{result['run_name']}_{ident}"
        out_path = out_dir / f"structure_{ident}.{fmt}"
        try:
            write_ase_structure(str(out_path), atoms)
            if not quiet:
                self._show_message(f"Saved: {out_path}")
            return True
        except Exception as exc:
            if not quiet:
                self._show_message(f"Save failed: {exc}")
            return False

    def _action_enter_search(self) -> None:
        if self._footer is None or self._main_frame is None:
            return
        self._footer.show_search()
        self._main_frame.focus_position = "footer"

    def _on_focus_change(self, result: dict | None) -> None:
        self._update_details(result)

    def _on_row_select(self, _table, result: dict) -> None:
        self._update_details(result)

    def keypress(self, size: tuple, key: str) -> str | None:
        if key in ("u", "U"):
            self._action_toggle_unconverged()
            return None
        if key in ("t", "T"):
            self._action_thickness()
            return None
        if key in ("d", "D"):
            self._action_toggle_duplicates()
            return None
        if key in ("s",):
            self._action_save()
            return None
        if key == "/":
            self._action_enter_search()
            return None
        if key in ("p", "P"):
            self._action_phonon()
            return None
        if key == "esc":
            if self._search_query:
                self._search_query = ""
                self._rebuild_table()
                self._show_message("")
                return None
            if self._phonon_task is not None:
                self._phonon_task.cancel()
            self._router.pop()
            return None
        return key

    def _action_phonon(self) -> None:
        if self._main_frame is None:
            return
        if not any(r.get("converged", True) for r in self._results):
            self._show_message(
                "No converged structures available for phonon calculation."
            )
            return

        from rapmat.tui.widgets.form import (
            FormGroup,
            checkbox_field,
            dropdown_field,
            float_field,
            int_field,
            tuple_field,
        )
        from rapmat.calculators import Calculators

        calc_options = [c.value for c in Calculators]

        form = FormGroup(
            fields=[
                int_field("top_n", "Top N structures", default=5),
                dropdown_field(
                    "apply_to",
                    "Apply to",
                    options=["Filtered view", "All converged"],
                    default=0,
                ),
                dropdown_field(
                    "calculator", "Calculator", options=calc_options, default=0
                ),
                tuple_field("supercell", "Supercell", size=3, default=(3, 3, 3)),
                tuple_field("mesh", "Q-point mesh", size=3, default=(20, 20, 20)),
                float_field("displacement", "Displacement", default=0.01),
                float_field("cutoff", "Imag freq cutoff", default=-0.15),
                checkbox_field("reduce_prim", "Reduce to primitive", default=True),
            ],
            label_width=18,
        )

        current_body = self._main_frame.body

        def _on_submit(_btn) -> None:
            errors = form.validate()
            if errors:
                err_text.set_text(("form_error", "; ".join(errors)))
                return
            vals = form.get_values()
            self._main_frame.body = current_body
            self._start_phonon_task(vals)

        def _on_cancel(_btn) -> None:
            self._main_frame.body = current_body

        err_text = urwid.Text("")
        submit_btn = urwid.AttrMap(
            urwid.Button("Run Phonons", on_press=_on_submit),
            None,
            focus_map="btn_focus",
        )
        cancel_btn = urwid.AttrMap(
            urwid.Button("Cancel", on_press=_on_cancel), None, focus_map="btn_focus"
        )
        btn_row = urwid.Columns(
            [("weight", 1, submit_btn), ("weight", 1, cancel_btn)], dividechars=2
        )

        dialog_body = urwid.Pile(
            [
                ("pack", urwid.Text(("section", " Phonon Calculation"), align="left")),
                ("pack", urwid.Divider("─")),
                ("pack", form),
                ("pack", urwid.Divider()),
                ("pack", err_text),
                ("pack", urwid.Divider()),
                ("pack", btn_row),
            ]
        )

        inner = urwid.LineBox(
            urwid.Padding(dialog_body, left=1, right=1),
            title="Phonon Parameters",
        )
        overlay = urwid.Overlay(
            urwid.AttrMap(inner, "dialog"),
            current_body,
            align=urwid.CENTER,
            width=(urwid.RELATIVE, 60),
            valign=urwid.MIDDLE,
            height=urwid.PACK,
            min_width=50,
        )
        self._main_frame.body = overlay

    def _start_phonon_task(self, vals: dict) -> None:
        if self._main_frame is None or self._body_pile is None:
            return

        from rapmat.calculators import Calculators
        from rapmat.tui.tasks import BackgroundTask
        from rapmat.tui.widgets.progress import ProgressPanel
        from rapmat.core.phonon_stability import compute_dynamical_stability_for_results

        top_n = int(vals.get("top_n", 5))
        calc_name = vals.get("calculator", Calculators.MATTERSIM.value)
        supercell = vals.get("supercell", (3, 3, 3))
        mesh = vals.get("mesh", (20, 20, 20))
        displacement = float(vals.get("displacement", 0.01))
        cutoff = float(vals.get("cutoff", -0.15))
        reduce_prim = bool(vals.get("reduce_prim", True))

        try:
            calc_enum = Calculators(calc_name)
        except ValueError:
            calc_enum = Calculators.MATTERSIM

        panel = ProgressPanel(title=" Phonon Calculation ")
        panel.set_progress(0, top_n)
        panel.add_log(f"Starting phonon calculation for top {top_n} structures...")

        self._body_pile.contents[1] = (
            urwid.BoxAdapter(panel, 12),
            self._body_pile.options("pack"),
        )

        apply_to = vals.get("apply_to", "Filtered view")
        if apply_to == "Filtered view":
            results_snapshot = list(self._get_display_results())
        else:
            results_snapshot = list(self._results)
        structures_snapshot = list(self._structures)
        store = self._state.store
        phonon_cutoff = (
            self._phonon_cutoff if self._phonon_cutoff is not None else cutoff
        )

        def _worker(progress) -> None:
            def _cb(
                current: int, total: int, message: str, is_log: bool = True
            ) -> None:
                if progress.cancelled:
                    raise KeyboardInterrupt("Cancelled by user")
                progress.update(current=current, total=total, message=message)
                if is_log:
                    progress.log(message)

            compute_dynamical_stability_for_results(
                results=results_snapshot,
                structures=structures_snapshot,
                phonon_top=top_n,
                phonon_cutoff=phonon_cutoff,
                phonon_supercell=supercell,
                phonon_mesh=mesh,
                phonon_displacement=displacement,
                phonon_calculator=calc_enum,
                store=store,
                progress_callback=_cb,
                symprec=self._get_symprec(),
                reduce_primitive=reduce_prim,
            )

        def _on_progress(current: int, total: int, message: str) -> None:
            panel.set_progress(current, total, message)

        def _on_log(line: str) -> None:
            panel.add_log(line)

        def _on_complete() -> None:
            panel.set_finished(True, "Phonon calculation complete.")
            self._show_dynamical_stability = True
            self._phonon_cutoff = phonon_cutoff
            self._on_phonon_complete(phonon_cutoff)
            self._rebuild_table()

            if self._body_pile is not None and self._details_panel is not None:
                self._body_pile.contents[1] = (
                    self._details_panel,
                    self._body_pile.options("pack"),
                )
            self._show_message("Phonon calculation complete.")

        def _on_error(error: str) -> None:
            panel.set_finished(False, f"Error: {error}")
            self._show_message(f"Phonon failed: {error}")

        loop = self._state.loop
        if loop is None:
            self._show_message("Cannot start background task: no event loop.")
            return

        self._phonon_task = BackgroundTask(
            fn=_worker,
            loop=loop,
            on_progress=_on_progress,
            on_log=_on_log,
            on_complete=_on_complete,
            on_error=_on_error,
        )
        self._phonon_task.start()
