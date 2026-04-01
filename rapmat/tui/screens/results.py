"""Results viewer screen for the Rapmat TUI."""

from ase import Atoms

from rapmat.tui.router import ScreenRouter
from rapmat.tui.state import AppState
from rapmat.tui.screens.base_results import BaseResultsScreen, _dyn_stability

class ResultsScreen(BaseResultsScreen):
    """CSP results viewer screen.

    Fetches relaxed structures for ``state.active_run`` from the store
    and displays them in an interactive table with a details panel.
    """

    title = "Results"

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        super().__init__(state, router)
        self._run_name: str = ""

    def update_footer_keys(self, message: str = "") -> None:
        if not self._state.status_bar:
            return
        keys = [
            ("u", "Unconverged"),
            ("t", "Thickness"),
            ("d", "Duplicates"),
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

    def _fetch_data(self) -> None:
        run_name = self._state.active_run or ""
        self._run_name = run_name

        store = self._state.store

        meta = store.get_run_metadata(run_name)
        config = meta.get("config", {}) if meta else {}
        self._pressure_gpa = float(config.get("pressure_gpa", 0.0))
        self._phonon_cutoff = config.get("phonon_cutoff")

        records = store.get_run_structures(run_name, status="relaxed")

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
                "duplicate": rec.get("duplicate"),
                "run_name": run_name,
            }
            if self._pressure_gpa > 0:
                entry["enthalpy_per_atom"] = rec.get("enthalpy_per_atom")
                entry["volume"] = rec.get("volume")
            self._results.append(entry)
            self._structures.append(rec["atoms"])

        self.title = (
            f"Results: {run_name} {self._pressure_gpa} GPa {len(self._results)} relaxed"
        )
        is_bulk = config.get("domain", "bulk") == "bulk"
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
        if self._show_duplicate_col:
            cols.append(("Dup", 5))
        cols.append(("Status", 8))
        return cols

    def _format_row(self, result: dict) -> list[str]:
        row = [
            str(result.get("id", "?")),
            str(result.get("formula", "N/A")),
            str(result.get("initial_spg", "N/A")),
            str(result.get("final_spg", "N/A")),
            f"{result.get('energy_per_atom', result.get('effective_per_atom', 0.0)):.4f}",
        ]
        if self._pressure_gpa > 0:
            h = result.get("enthalpy_per_atom")
            row.append(f"{h:.4f}" if h is not None else "N/A")
        
        fmax = result.get('fmax')
        row.append(f"{fmax:.3f}" if fmax is not None else "N/A")
        
        if self._show_thickness:
            t = result.get("thickness")
            row.append("" if t is None else f"{t:.2f}")
        if self._show_dynamical_stability:
            dyn = _dyn_stability(result, self._phonon_cutoff)
            row.append("Yes" if dyn is True else ("No" if dyn is False else "N/A"))
        if self._show_duplicate_col:
            dup = result.get("duplicate")
            row.append("Yes" if dup is True else ("No" if dup is False else ""))
        
        conv = result.get("converged")
        row.append("OK" if conv is True or conv is None else "Unconv")
        return row

    def _get_symprec(self) -> float:
        meta = self._state.store.get_run_metadata(self._run_name)
        if meta and "config" in meta:
            return meta["config"].get("symprec", 1e-3)
        return 1e-3

    def _on_phonon_complete(self, phonon_cutoff: float) -> None:
        meta = self._state.store.get_run_metadata(self._run_name)
        if meta:
            config = meta.get("config", {})
            config["phonon_cutoff"] = phonon_cutoff
            self._state.store.update_run_config(self._run_name, config)

    def keypress(self, size: tuple, key: str) -> str | None:
        # Capital S also saves (base only binds lowercase s)
        if key == "S":
            self._action_save()
            return None
        return super().keypress(size, key)

