"""Phase analysis screen for the Rapmat TUI."""

from pathlib import Path

from rapmat.tui.router import ScreenRouter
from rapmat.tui.state import AppState
from rapmat.tui.screens.base_results import BaseResultsScreen, _dyn_stability
from rapmat.tui.widgets.dialog import ModalDialog


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

    def update_footer_keys(self, message: str = "") -> None:
        if not self._state.status_bar:
            return
        keys = [
            ("a", "Show Best" if self._show_all else "Show All"),
            ("u", "Unconverged"),
            ("t", "Thickness"),
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

    def _fetch_data(self) -> None:
        study_id = self._state.active_study
        if not study_id:
            self.title = "No active study selected."
            return
        if isinstance(study_id, str) and ":" in study_id:
            study_id = study_id.split(":")[-1]

        self._study_id = str(study_id)
        store = self._state.store
        study = store.get_study(self._study_id)
        if not study:
            self.title = f"Study '{self._study_id}' not found."
            return

        symprec = study.get("config", {}).get("symprec", 1e-3)
        self._phonon_cutoff = study.get("config", {}).get("phonon_cutoff")

        self._study_system = study.get("system", "")
        from rapmat.utils.common import parse_system

        elements = parse_system(self._study_system)
        self._system_size = len(elements)

        if self._system_size < 2:
            from rapmat.core.hull import build_energy_ranking

            sd = build_energy_ranking(store, self._study_id)
            self._use_enthalpy = False
        else:
            from rapmat.core.hull import build_phase_diagram

            _, sd, use_eth = build_phase_diagram(
                store, self._study_id, symprec=symprec, show_all=self._show_all
            )
            self._use_enthalpy = use_eth

        # Map to expected structure_data for BaseResultsScreen
        self._results = []
        self._structures = []
        for i, data in enumerate(sd):
            # merge in ID and structure index
            data["id"] = i + 1
            data["structure_index"] = i
            data["converged"] = True  # Only relaxed ones are shown here
            self._results.append(data)
            if "atoms" in data:
                self._structures.append(data["atoms"])

        self.title = f"Phase Analysis: {self._study_system} | {len(self._results)} structures"
        self._show_thickness = any(
            r.get("thickness") is not None for r in self._results
        )
        self._show_dynamical_stability = any(
            r.get("min_phonon_freq") is not None or "dynamical_stability" in r
            for r in self._results
        )
        self._show_duplicate_col = False  # Not really used at study level

    def _columns_def(self) -> list[tuple[str, int]]:
        if self._system_size < 2:
            return [
                ("Formula", 14),
                ("E/atom", 10),
                ("Run", 22),
            ]
        elif self._system_size == 2:
            cols = [
                ("Formula", 14),
                ("x", 8),
                ("E/atom", 10),
                ("E_form", 10),
                ("EAH", 10),
            ]
            if self._show_dynamical_stability:
                cols.append(("Dyn", 5))
            cols.append(("Run", 22))
            return cols
        else:
            cols = [
                ("Formula", 14),
                ("E/atom", 10),
                ("E_form", 10),
                ("EAH", 10),
            ]
            if self._show_dynamical_stability:
                cols.append(("Dyn", 5))
            cols.append(("Run", 22))
            return cols

    def _format_row(self, result: dict) -> list[str]:
        formula = result.get("reduced_formula", result.get("formula", "N/A"))
        epa = result.get("energy_per_atom", result.get("effective_per_atom", 0.0))
        run = result.get("run_name", "N/A")

        if self._system_size < 2:
            return [formula, f"{epa:.4f}", run]
        elif self._system_size == 2:
            x = f"{result.get('composition_frac', 0.0):.3f}"
            e_form = f"{result.get('formation_energy', 0.0):.4f}"
            eah = f"{result.get('energy_above_hull', 0.0):.4f}"
            row = [formula, x, f"{epa:.4f}", e_form, eah]
            if self._show_dynamical_stability:
                dyn = _dyn_stability(result, self._phonon_cutoff)
                row.append("Yes" if dyn is True else ("No" if dyn is False else ""))
            row.append(run)
            return row
        else:
            e_form = f"{result.get('formation_energy', 0.0):.4f}"
            eah = f"{result.get('energy_above_hull', 0.0):.4f}"
            row = [formula, f"{epa:.4f}", e_form, eah]
            if self._show_dynamical_stability:
                dyn = _dyn_stability(result, self._phonon_cutoff)
                row.append("Yes" if dyn is True else ("No" if dyn is False else ""))
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
            self._fetch_data()
            self._rebuild_table()
            self.update_footer_keys(
                f"{'Showing all' if self._show_all else 'Showing best'}"
            )
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
