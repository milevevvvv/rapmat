"""Phase analysis screen for the Rapmat TUI."""

from pathlib import Path

import urwid

from rapmat.tui.router import ScreenRouter
from rapmat.tui.state import AppState
from rapmat.tui.screens.base_results import BaseResultsScreen, _dyn_stability
from rapmat.tui.tasks import BackgroundTask
from rapmat.tui.widgets.dialog import ModalDialog
from rapmat.tui.widgets.progress import ProgressPanel


class PhaseAnalysisScreen(BaseResultsScreen):
    """Phase analysis viewer — supports unary, binary, and ternary+ systems."""

    title = "Phase Analysis"

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        super().__init__(state, router)
        self._study_id: str = ""
        self._study_system: str = ""
        self._system_size: int = 0
        self._use_enthalpy: bool = False
        self._show_all: bool = False
        self._loading_task: BackgroundTask | None = None
        self._outer_placeholder: urwid.WidgetPlaceholder | None = None

    def update_footer_keys(self, message: str = "") -> None:
        if not self._state.status_bar:
            return
        keys = [
            ("a", "Show Best" if self._show_all else "Show All"),
            ("d", "Duplicates"),
            ("t", "Thickness" if self._show_thickness else ""),
            ("/", "Search"),
            ("s", "Save"),
            ("p", "Phonons"),
            ("S", "Save Plot" if self._system_size == 2 else ""),
            ("Esc", "Back"),
        ]
        # Filter out empty keys
        keys = [k for k in keys if k[1]]
        self._state.status_bar.set_keys(keys)
        if message:
            self._state.status_bar.set_message(message)
        else:
            self._state.status_bar.clear_message()

    # ------------------------------------------------------------------ #
    #  Async build: show loading panel, compute in background, then table
    # ------------------------------------------------------------------ #

    def build(self) -> urwid.Widget:
        self._outer_placeholder = urwid.WidgetPlaceholder(urwid.SolidFill(" "))
        self.update_footer_keys()
        self._start_async_fetch()
        return self._outer_placeholder

    def _start_async_fetch(self) -> None:
        """Run phase analysis computation in a background thread."""
        panel = ProgressPanel(title=" Phase Analysis ")
        panel.set_progress(0, 1, "Computing phase analysis...")
        panel.add_log("Loading structures from database...")

        # Show the progress panel centered
        loading_widget = urwid.Filler(urwid.BoxAdapter(panel, 10), valign="middle")
        if self._outer_placeholder is not None:
            self._outer_placeholder.original_widget = loading_widget

        loop = self._state.loop
        if loop is None:
            # Fallback: synchronous fetch
            self._fetch_data()
            self._main_frame = self._build_frame()
            if self._outer_placeholder is not None:
                self._outer_placeholder.original_widget = self._main_frame
            return

        # Capture state needed by the worker
        _store = self._state.store
        _active_study = self._state.active_study
        _show_all = self._show_all

        # Mutable container for results from the worker thread
        _result_box: dict = {}

        def _worker(progress) -> None:
            progress.log("Resolving study...")
            study_id = _active_study
            if not study_id:
                progress.fail("No active study selected.")
                return
            if isinstance(study_id, str) and ":" in study_id:
                study_id = study_id.split(":")[-1]

            study = _store.get_study(str(study_id))
            if not study:
                progress.fail(f"Study '{study_id}' not found.")
                return

            symprec = study.get("config", {}).get("symprec", 1e-3)
            system = study.get("system", "")
            from rapmat.utils.common import parse_system
            elements = parse_system(system)
            system_size = len(elements)

            progress.log(f"System: {system} ({system_size} element{'s' if system_size != 1 else ''})")

            if system_size < 2:
                from rapmat.core.hull import build_energy_ranking
                progress.log("Building energy ranking...")
                sd = build_energy_ranking(_store, str(study_id))
                use_enthalpy = False
            else:
                from rapmat.core.hull import build_phase_diagram
                progress.log("Building phase diagram...")
                _, sd, use_enthalpy = build_phase_diagram(
                    _store, str(study_id), symprec=symprec, show_all=_show_all
                )

            progress.update(1, 1, f"Done — {len(sd)} structures")
            progress.log(f"Computed {len(sd)} structures.")

            _result_box["study_id"] = str(study_id)
            _result_box["study"] = study
            _result_box["system"] = system
            _result_box["system_size"] = system_size
            _result_box["sd"] = sd
            _result_box["use_enthalpy"] = use_enthalpy

        def _on_progress(current: int, total: int, message: str) -> None:
            panel.set_progress(current, total, message)

        def _on_log(line: str) -> None:
            panel.add_log(line)

        def _on_complete() -> None:
            self._apply_fetch_result(_result_box)
            self._main_frame = self._build_frame()
            if self._outer_placeholder is not None:
                self._outer_placeholder.original_widget = self._main_frame

        def _on_error(error: str) -> None:
            panel.set_finished(False, f"Error: {error}")

        self._loading_task = BackgroundTask(
            fn=_worker,
            loop=loop,
            on_progress=_on_progress,
            on_log=_on_log,
            on_complete=_on_complete,
            on_error=_on_error,
        )
        self._loading_task.start()

    def _apply_fetch_result(self, box: dict) -> None:
        """Apply results from the background worker to self."""
        self._study_id = box.get("study_id", "")
        self._study_system = box.get("system", "")
        self._system_size = box.get("system_size", 0)
        self._use_enthalpy = box.get("use_enthalpy", False)

        study = box.get("study", {})
        self._phonon_cutoff = study.get("config", {}).get("phonon_cutoff")

        sd = box.get("sd", [])
        self._results = []
        self._structures = []
        for i, data in enumerate(sd):
            data["id"] = i + 1
            data["structure_index"] = i
            data["converged"] = True
            self._results.append(data)
            if "atoms" in data:
                self._structures.append(data["atoms"])

        self.title = f"Phase Analysis: {self._study_system} | {len(self._results)} structures"
        is_bulk = study.get("domain", "bulk") == "bulk"
        self._show_thickness = (not is_bulk) and any(
            r.get("thickness") is not None for r in self._results
        )
        self._show_dynamical_stability = any(
            r.get("min_phonon_freq") is not None or "dynamical_stability" in r
            for r in self._results
        )
        self._show_duplicate_col = any(
            r.get("duplicate") is not None for r in self._results
        )

    def _fetch_data(self) -> None:
        """Synchronous fallback (used when event loop is not available)."""
        study_id = self._state.active_study
        if not study_id:
            return
        if isinstance(study_id, str) and ":" in study_id:
            study_id = study_id.split(":")[-1]

        store = self._state.store
        study = store.get_study(str(study_id))
        if not study:
            return

        box: dict = {}
        symprec = study.get("config", {}).get("symprec", 1e-3)
        system = study.get("system", "")
        from rapmat.utils.common import parse_system
        elements = parse_system(system)
        system_size = len(elements)

        if system_size < 2:
            from rapmat.core.hull import build_energy_ranking
            sd = build_energy_ranking(store, str(study_id))
            use_enthalpy = False
        else:
            from rapmat.core.hull import build_phase_diagram
            _, sd, use_enthalpy = build_phase_diagram(
                store, str(study_id), symprec=symprec, show_all=self._show_all
            )

        box.update({
            "study_id": str(study_id), "study": study,
            "system": system, "system_size": system_size,
            "sd": sd, "use_enthalpy": use_enthalpy,
        })
        self._apply_fetch_result(box)

    def _columns_def(self) -> list[tuple[str, int]]:
        # Core columns per system size
        if self._system_size < 2:
            cols = [
                ("Formula", 14),
                ("Final SG", 12),
                ("E/atom", 10),
            ]
        elif self._system_size == 2:
            cols = [
                ("Formula", 14),
                ("Final SG", 12),
                ("x", 8),
                ("E/atom", 10),
                ("E_form", 10),
                ("EAH", 10),
            ]
        else:
            cols = [
                ("Formula", 14),
                ("Final SG", 12),
                ("E/atom", 10),
                ("E_form", 10),
                ("EAH", 10),
            ]
        # Optional columns (shared across all system sizes)
        if self._show_thickness:
            cols.append(("Thick(A)", 9))
        if self._show_dynamical_stability:
            cols.append(("Dyn", 5))
        if self._show_duplicate_col:
            cols.append(("Dup", 5))
        cols.append(("Run", 22))
        return cols

    def _format_row(self, result: dict) -> list[str]:
        formula = result.get("formula", result.get("reduced_formula", "N/A"))
        epa = result.get("energy_per_atom", result.get("effective_per_atom", 0.0))
        run = result.get("run_name", "N/A")
        spg = str(result.get("final_spg", ""))

        if self._system_size < 2:
            row = [formula, spg, f"{epa:.4f}"]
        elif self._system_size == 2:
            x = f"{result.get('composition_frac', 0.0):.3f}"
            e_form = f"{result.get('formation_energy', 0.0):.4f}"
            eah = f"{result.get('energy_above_hull', 0.0):.4f}"
            row = [formula, spg, x, f"{epa:.4f}", e_form, eah]
        else:
            e_form = f"{result.get('formation_energy', 0.0):.4f}"
            eah = f"{result.get('energy_above_hull', 0.0):.4f}"
            row = [formula, spg, f"{epa:.4f}", e_form, eah]

        # Optional columns (must match _columns_def order)
        if self._show_thickness:
            t = result.get("thickness")
            row.append("" if t is None else f"{t:.2f}")
        if self._show_dynamical_stability:
            dyn = _dyn_stability(result, self._phonon_cutoff)
            row.append("Yes" if dyn is True else ("No" if dyn is False else ""))
        if self._show_duplicate_col:
            dup = result.get("duplicate")
            row.append("Yes" if dup is True else ("No" if dup is False else ""))
        row.append(run)
        return row

    def _get_symprec(self) -> float:
        study = self._state.store.get_study(self._study_id)
        if study and "config" in study:
            return study["config"].get("symprec", 1e-3)
        return 1e-3

    def _get_extra_details(self, result: dict) -> list:
        extra: list = []
        if self._system_size >= 2:
            if "formation_energy" in result:
                extra.append(
                    ("details", f"Formation Energy: {result['formation_energy']:.4f} eV/atom\n")
                )
            if "energy_above_hull" in result:
                extra.append(
                    ("details", f"Energy Above Hull: {result['energy_above_hull']:.4f} eV/atom\n")
                )
            if "is_stable" in result:
                extra.append(
                    ("details", f"Hull Stable: {'Yes' if result['is_stable'] else 'No'}\n")
                )
        return extra

    def _on_phonon_complete(self, phonon_cutoff: float) -> None:
        study = self._state.store.get_study(self._study_id)
        if study:
            config = study.get("config", {})
            config["phonon_cutoff"] = phonon_cutoff
            self._state.store.update_study(self._study_id, {"config": config})
            runs = self._state.store.get_study_runs(self._study_id)
            for r in runs:
                run_config = r.get("config", {})
                run_config["phonon_cutoff"] = phonon_cutoff
                self._state.store.update_run_config(r["name"], run_config)

    def keypress(self, size: tuple, key: str) -> str | None:
        if key in ("a", "A"):
            self._show_all = not self._show_all
            self._start_async_fetch()
            return None
        if key in ("S",) and self._system_size == 2:
            self._open_save_plot_modal()
            return None
        return super().keypress(size, key)

    def _open_save_plot_modal(self) -> None:
        if self._system_size != 2 or self._main_frame is None:
            return

        current_body = self._main_frame.body

        def _do_save(path_str: str) -> None:
            self._main_frame.body = current_body
            plot_path = Path(path_str)
            try:
                from rapmat.core.hull import plot_binary_hull

                plot_binary_hull(
                    self._results,
                    self._study_system,
                    save_path=plot_path,
                    show=False,
                    use_enthalpy=self._use_enthalpy,
                )
                self._show_message(f"Plot saved to {plot_path}")
            except Exception as exc:
                self._show_message(f"Save failed: {exc}")

        def _cancel() -> None:
            self._main_frame.body = current_body

        dlg = ModalDialog.input_text(
            title="Save Hull Plot",
            message="Enter file path for the hull plot:",
            parent=current_body,
            on_save=_do_save,
            on_cancel=_cancel,
            default="hull.png",
        )
        self._main_frame.body = dlg
