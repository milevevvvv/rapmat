"""Results viewer screen for the Rapmat TUI."""

import math
from pathlib import Path
from typing import Optional

import urwid
from ase import Atoms
from ase.io import write as write_ase_structure

from rapmat.tui.widgets.table import SelectableRow, SortableTable

from rapmat.tui.router import ScreenRouter
from rapmat.tui.state import AppState
from rapmat.tui.tasks import BackgroundTask

# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


def _dyn_stability(result: dict, phonon_cutoff: float | None) -> Optional[bool]:
    """Derive dynamical stability from min_phonon_freq + cutoff."""
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
    if not result.get("converged"):
        return "unconv"
    dyn = _dyn_stability(result, phonon_cutoff)
    if dyn is False:
        return "error"
    return "body"


# ------------------------------------------------------------------ #
#  Footer widget (search / status)
# ------------------------------------------------------------------ #


class _ResultsFooter(urwid.WidgetWrap):
    def __init__(self, screen: "ResultsScreen") -> None:
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


# ------------------------------------------------------------------ #
#  Save dialog (format + path picker)
# ------------------------------------------------------------------ #


class _SaveDialog(urwid.WidgetWrap):
    """Inline save dialog overlaid on the results frame body."""

    signals = ["close"]

    def __init__(self, parent: urwid.Widget, on_save) -> None:
        self._on_save = on_save

        self._fmt_group: list = []
        cif_rb = urwid.RadioButton(self._fmt_group, "cif")
        xyz_rb = urwid.RadioButton(self._fmt_group, "xyz")
        self._dir_edit = urwid.Edit(caption="Directory: ", edit_text=str(Path.cwd()))
        self._standardize_cb = urwid.CheckBox("Standardize cell", state=False)

        def _ok(_btn: urwid.Button) -> None:
            fmt = next((rb.label for rb in self._fmt_group if rb.state), "cif")
            directory = self._dir_edit.edit_text.strip() or str(Path.cwd())
            standardize = self._standardize_cb.state
            self._emit("close", True)
            on_save(fmt, directory, standardize)

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
            min_width=40,
        )
        super().__init__(overlay)

    def keypress(self, size: tuple, key: str) -> str | None:
        if key == "esc":
            self._emit("close", False)
            return None
        return super().keypress(size, key)


# ------------------------------------------------------------------ #
#  Main screen
# ------------------------------------------------------------------ #


class ResultsScreen:
    """CSP results viewer screen.

    Fetches relaxed structures for ``state.active_run`` from the store
    and displays them in an interactive table with a details panel.
    """

    title = "Results"

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        self._state = state
        self._router = router

        # Run-level config (loaded in _fetch_data)
        self._run_name: str = ""
        self._pressure_gpa: float = 0.0
        self._thickness_cutoff: float | None = None
        self._phonon_cutoff: float | None = None

        # Display state
        self._results: list[dict] = []
        self._structures: list[Atoms] = []
        self._hide_unconverged: bool = True
        self._hide_thick: bool = False
        self._search_query: str = ""
        self._show_thickness: bool = False
        self._show_dynamical_stability: bool = False
        self._app_message: str = ""

        # Urwid widgets (set in build)
        self._main_frame: urwid.Frame | None = None
        self._table: SortableTable | None = None
        self._details_text: urwid.Text | None = None
        self._footer: _ResultsFooter | None = None
        self._details_panel: urwid.Widget | None = None
        self._body_pile: urwid.Pile | None = None

        # Background task for phonon calculation
        self._phonon_task: "BackgroundTask | None" = None

    # ------------------------------------------------------------------ #
    #  Screen protocol
    # ------------------------------------------------------------------ #

    def build(self) -> urwid.Widget:
        self._fetch_data()
        self._main_frame = self._build_frame()
        return self._main_frame

    def on_resume(self) -> None:
        self.update_footer_keys()

    def on_leave(self) -> None:
        pass

    def update_footer_keys(self, message: str = "") -> None:
        if not self._state.status_bar:
            return
        keys = [
            ("u", "Unconverged"),
            ("t", "Thickness"),
            ("/", "Search"),
            ("s", "Save"),
            ("p", "Phonons"),
            ("Esc", "Back"),
        ]
        self._state.status_bar.set_keys(keys)
        if message:
            self._state.status_bar.set_message(message)
        else:
            self._state.status_bar.clear_message()

    # ------------------------------------------------------------------ #
    #  Data loading
    # ------------------------------------------------------------------ #

    def _fetch_data(self) -> None:
        run_name = self._state.active_run or ""
        self._run_name = run_name

        store = self._state.store

        meta = store.get_run_metadata(run_name)
        config = meta.get("config", {}) if meta else {}
        self._pressure_gpa = float(config.get("pressure_gpa", 0.0))
        self._phonon_cutoff = config.get("phonon_cutoff")

        records = store.get_run_structures(run_name, status="relaxed")

        # Sort by enthalpy when under pressure, otherwise by energy
        if self._pressure_gpa > 0:
            records.sort(
                key=lambda r: (
                    r.get("enthalpy_per_atom")
                    if r.get("enthalpy_per_atom") is not None
                    else r["energy_per_atom"]
                )
            )
        else:
            records.sort(key=lambda r: r["energy_per_atom"])

        self._results = []
        self._structures = []
        for idx, rec in enumerate(records):
            entry: dict = {
                "id": idx + 1,
                "structure_index": idx,
                "structure_id": rec["id"],
                "formula": rec["formula"],
                "initial_spg": rec.get("initial_spg", "N/A"),
                "final_spg": rec.get("final_spg", "N/A"),
                "energy_per_atom": rec["energy_per_atom"],
                "fmax": rec["fmax"],
                "converged": rec["converged"],
                "thickness": rec.get("thickness") or None,
                "min_phonon_freq": rec.get("min_phonon_freq"),
            }
            if self._pressure_gpa > 0:
                entry["enthalpy_per_atom"] = rec.get("enthalpy_per_atom")
                entry["volume"] = rec.get("volume")
            self._results.append(entry)
            self._structures.append(rec["atoms"])

        self.title = (
            f"Results: {run_name} {self._pressure_gpa} GPa {len(self._results)} relaxed"
        )
        self._show_thickness = any(
            r.get("thickness") is not None for r in self._results
        )
        self._show_dynamical_stability = any(
            r.get("min_phonon_freq") is not None or "dynamical_stability" in r
            for r in self._results
        )

    # ------------------------------------------------------------------ #
    #  Layout
    # ------------------------------------------------------------------ #

    def _columns_def(self) -> list[tuple[str, int]]:
        cols = [
            ("ID", 6),
            ("Formula", 14),
            ("Init SG", 12),
            ("Final SG", 12),
            ("E/atom", 10),
        ]
        if self._pressure_gpa > 0:
            cols.append(("H/atom", 10))
        cols.append(("Fmax", 8))
        if self._show_thickness:
            cols.append(("Thick(A)", 9))
        if self._show_dynamical_stability:
            cols.append(("Dyn.stable", 11))
        cols.append(("Status", 8))
        return cols

    def _format_row(self, result: dict) -> list[str]:
        row = [
            str(result.get("id", "?")),
            str(result.get("formula", "N/A")),
            str(result.get("initial_spg", "N/A")),
            str(result.get("final_spg", "N/A")),
            f"{result.get('energy_per_atom', 0.0):.4f}",
        ]
        if self._pressure_gpa > 0:
            h = result.get("enthalpy_per_atom")
            row.append(f"{h:.4f}" if h is not None else "N/A")
        row.append(f"{result.get('fmax', 0.0):.3f}")
        if self._show_thickness:
            t = result.get("thickness")
            row.append("" if t is None else f"{t:.2f}")
        if self._show_dynamical_stability:
            dyn = _dyn_stability(result, self._phonon_cutoff)
            row.append("Yes" if dyn is True else ("No" if dyn is False else "N/A"))
        row.append("OK" if result.get("converged") else "Unconv")
        return row

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

        self._details_text = urwid.Text("", align="left")
        self._details_panel = urwid.BoxAdapter(
            urwid.LineBox(
                urwid.Filler(self._details_text, valign="top"),
                title="Structure Details",
            ),
            14,
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

        # Populate details for the initially focused row
        self._update_details(self._table.get_focused_row())

        self.update_footer_keys()

        return frame

    # ------------------------------------------------------------------ #
    #  Display helpers
    # ------------------------------------------------------------------ #

    def _get_display_results(self) -> list[dict]:
        res = self._results
        if self._hide_unconverged:
            res = [r for r in res if r.get("converged")]
        if self._hide_thick and self._thickness_cutoff is not None:
            res = [
                r
                for r in res
                if r.get("thickness") is not None
                and r["thickness"] <= self._thickness_cutoff
            ]
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
        # Columns may change if pressure/thickness/phonon flags changed
        new_cols = self._columns_def()
        self._table.update_columns(new_cols)
        self._table.set_data(self._get_display_results())

    def _update_details(self, result: dict | None) -> None:
        if self._details_text is None:
            return

        markup: list = []

        if result is None:
            if (
                self._hide_unconverged
                and self._results
                and not any(r.get("converged") for r in self._results)
            ):
                markup.append(
                    (
                        "details",
                        "All structures are unconverged. Press [u] to show them.\n",
                    )
                )
            else:
                markup.append(("details", "No structure selected.\n"))
        else:
            idx = result.get("structure_index")
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                idx = None

            if idx is not None and 0 <= idx < len(self._structures):
                atoms = self._structures[idx]
                cell_lengths = atoms.get_cell().lengths()

                markup.append(("details_title", f"ID: {result.get('id', idx + 1)}\n"))
                markup.append(("details", f"Formula: {atoms.get_chemical_formula()}\n"))
                markup.append(("details", f"Atoms: {len(atoms)}\n"))
                markup.append(
                    (
                        "details",
                        f"Cell (Å): {cell_lengths[0]:.3f}, "
                        f"{cell_lengths[1]:.3f}, {cell_lengths[2]:.3f}\n",
                    )
                )
                markup.append(
                    ("details", f"Initial SG: {result.get('initial_spg', 'N/A')}\n")
                )
                markup.append(
                    ("details", f"Final SG: {result.get('final_spg', 'N/A')}\n")
                )
                markup.append(
                    (
                        "details",
                        f"Energy/atom: {result.get('energy_per_atom', 0.0):.4f} eV\n",
                    )
                )
                if self._pressure_gpa > 0:
                    h = result.get("enthalpy_per_atom")
                    if h is not None:
                        markup.append(("details", f"Enthalpy/atom: {h:.4f} eV\n"))
                    vol = result.get("volume")
                    if vol is not None:
                        markup.append(("details", f"Volume: {vol:.3f} Å³\n"))
                    markup.append(("details", f"Pressure: {self._pressure_gpa} GPa\n"))
                markup.append(("details", f"Fmax: {result.get('fmax', 0.0):.3f}\n"))
                markup.append(
                    ("details", f"Converged: {bool(result.get('converged'))}\n")
                )
                t = result.get("thickness")
                if t is not None:
                    markup.append(("details", f"Thickness (Å): {t:.2f}\n"))
                if self._show_dynamical_stability:
                    dyn = _dyn_stability(result, self._phonon_cutoff)
                    dyn_str = (
                        "Yes" if dyn is True else ("No" if dyn is False else "N/A")
                    )
                    markup.append(("details", f"Dynamical stability: {dyn_str}\n"))
                min_freq = result.get("min_phonon_freq")
                if min_freq is not None:
                    try:
                        f = float(min_freq)
                        if not math.isnan(f):
                            markup.append(
                                ("details", f"Min phonon freq (THz): {f:.4f}\n")
                            )
                    except (TypeError, ValueError):
                        pass
            else:
                markup.append(
                    ("details", "No structure data available for this row.\n")
                )

        self._details_text.set_text(markup)

    # ------------------------------------------------------------------ #
    #  Actions
    # ------------------------------------------------------------------ #

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

    def _action_thickness(self) -> None:
        """Show thickness filter dialog, then apply dynamic filter."""
        if self._main_frame is None:
            return
        if not self._show_thickness:
            self._show_message("No thickness data available for this run.")
            return

        from rapmat.tui.widgets.form import FormGroup, float_field

        default_val = self._thickness_cutoff if self._thickness_cutoff is not None else 0.0
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
            self._main_frame.body = current_body  # type: ignore[union-attr]
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
            self._main_frame.body = current_body  # type: ignore[union-attr]

        def _on_clear(_btn) -> None:
            self._main_frame.body = current_body  # type: ignore[union-attr]
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
        idx = result.get("structure_index")
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            idx = None
        if idx is None or idx < 0 or idx >= len(self._structures):
            self._show_message("No structure data available to save.")
            return

        current_body = self._main_frame.body

        def _on_save(fmt: str, directory: str, standardize: bool) -> None:
            self._main_frame.body = current_body  # type: ignore[union-attr]
            self._do_save(result, idx, fmt, directory, standardize)

        def _on_cancel() -> None:
            self._main_frame.body = current_body  # type: ignore[union-attr]

        save_dlg = _SaveDialog(current_body, _on_save)
        urwid.connect_signal(
            save_dlg, "close", lambda _w, ok: _on_cancel() if not ok else None
        )
        self._main_frame.body = save_dlg

    def _do_save(self, result: dict, idx: int, fmt: str, directory: str, standardize: bool = True) -> None:
        from rapmat.utils.structure import standardize_atoms

        atoms = self._structures[idx]
        if standardize:
            atoms = standardize_atoms(atoms)
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        ident = str(result.get("id", idx + 1)).replace("/", "_")
        out_path = out_dir / f"structure_{ident}.{fmt}"
        try:
            write_ase_structure(str(out_path), atoms)
            self._show_message(f"Saved: {out_path}")
        except Exception as exc:
            self._show_message(f"Save failed: {exc}")

    def _action_enter_search(self) -> None:
        if self._footer is None or self._main_frame is None:
            return
        self._footer.show_search()
        self._main_frame.focus_position = "footer"

    # ------------------------------------------------------------------ #
    #  Callbacks
    # ------------------------------------------------------------------ #

    def _on_focus_change(self, result: dict | None) -> None:
        self._update_details(result)

    def _on_row_select(self, _table, result: dict) -> None:
        # Enter on a row -- same as focus change for now (no drill-down in Phase 1)
        self._update_details(result)

    def keypress(self, size: tuple, key: str) -> str | None:
        if key in ("u", "U"):
            self._action_toggle_unconverged()
            return None
        if key in ("t", "T"):
            self._action_thickness()
            return None
        if key in ("s", "S"):
            self._action_save()
            return None
        if key == "/":
            self._action_enter_search()
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
        if key in ("p", "P"):
            self._action_phonon()
            return None
        return key

    # ------------------------------------------------------------------ #
    #  Phonon action
    # ------------------------------------------------------------------ #

    def _action_phonon(self) -> None:
        """Show phonon parameter dialog, then run computation in background."""
        if self._main_frame is None:
            return
        if not any(r.get("converged") for r in self._results):
            self._show_message(
                "No converged structures available for phonon calculation."
            )
            return

        from rapmat.tui.widgets.dropdown import DropdownSelect
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
            self._main_frame.body = current_body  # type: ignore[union-attr]
            self._start_phonon_task(vals)

        def _on_cancel(_btn) -> None:
            self._main_frame.body = current_body  # type: ignore[union-attr]

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
        """Launch the phonon computation in a background thread."""
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

        # Replace the details panel with the progress panel
        self._body_pile.contents[1] = (
            urwid.BoxAdapter(panel, 12),
            self._body_pile.options("pack"),
        )

        results_snapshot = list(self._results)
        structures_snapshot = list(self._structures)
        store = self._state.store
        phonon_cutoff = (
            self._phonon_cutoff if self._phonon_cutoff is not None else cutoff
        )
        meta = store.get_run_metadata(self._run_name)
        config = meta.get("config", {}) if meta else {}

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
                symprec=config.get("symprec", 1e-3) if config else 1e-3,
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
            meta = store.get_run_metadata(self._run_name)
            if meta:
                config = meta.get("config", {})
                config["phonon_cutoff"] = phonon_cutoff
                store.update_run_config(self._run_name, config)
            self._rebuild_table()
            # Restore details panel
            if self._body_pile is not None and self._details_panel is not None:
                self._body_pile.contents[1] = (
                    self._details_panel,
                    self._body_pile.options("pack"),
                )
            self._show_message("Phonon calculation complete.")

        def _on_error(error: str) -> None:
            panel.set_finished(False, f"Error: {error}")
            self._show_message(f"Phonon failed: {error}")

        # Resolve the main loop from AppState (set by RapmatApp on startup)
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
