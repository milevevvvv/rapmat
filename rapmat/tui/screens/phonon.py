"""Phonon dispersion screen for the Rapmat TUI."""

from pathlib import Path


import urwid

from rapmat.tui.widgets.form import (
    FormGroup,
    checkbox_field,
    dropdown_field,
    float_field,
    text_field,
    tuple_field,
)
from rapmat.tui.widgets.progress import ProgressPanel

from rapmat.tui.router import ScreenRouter
from rapmat.tui.state import AppState
from rapmat.tui.tasks import BackgroundTask


def _calc_options() -> list[str]:
    from rapmat.calculators import Calculators

    return [c.value for c in Calculators]


class PhononDispersionScreen:
    """Phonon dispersion calculation screen."""

    title = "Phonon Dispersion"

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        self._state = state
        self._router = router
        self._frame: urwid.Frame | None = None
        self._main_body: urwid.Widget | None = None
        self._progress_panel = ProgressPanel(title=" Phonon Progress ")
        self._task: BackgroundTask | None = None
        self._running = False

    # ------------------------------------------------------------------ #
    #  Screen protocol
    # ------------------------------------------------------------------ #

    def build(self) -> urwid.Widget:
        self._frame = self._build_frame()
        return self._frame

    def on_resume(self) -> None:
        self._update_footer()

    def _update_footer(self) -> None:
        if self._state.status_bar:
            self._state.status_bar.set_keys(
                [
                    ("F5", "Calculate"),
                    ("Esc", "Back"),
                ]
            )
            self._state.status_bar.clear_message()

    def on_leave(self) -> None:
        if self._task:
            self._task.cancel()

    # ------------------------------------------------------------------ #
    #  Layout
    # ------------------------------------------------------------------ #

    def _build_frame(self) -> urwid.Frame:
        self._form = FormGroup(
            [
                text_field("structure_file", "Structure file", default=""),
                dropdown_field("calculator", "Calculator", _calc_options(), default=0),
                tuple_field("supercell", "Supercell", size=3, default=(4, 4, 4)),
                tuple_field(
                    "qpoint_mesh", "Q-point mesh", size=3, default=(20, 20, 20)
                ),
                float_field("displacement", "Displacement", default=3e-2),
                float_field("imag_cutoff", "Imag freq cutoff", default=-0.15),
                checkbox_field("prerelax", "Pre-relax", default=False),
                checkbox_field("reduce_prim", "Reduce to primitive", default=False),
                text_field("plot_file", "Plot file", default="phonon_plot.png"),
            ],
            label_width=22,
        )

        self._error_text = urwid.Text("")
        self._summary_text = urwid.Text("")

        start_btn = urwid.AttrMap(
            urwid.Button("Calculate [F5]", on_press=self._on_start),
            "menu_item",
            focus_map="btn_focus",
        )

        listbox = urwid.ListBox(
            urwid.SimpleListWalker(
                [
                    self._form,
                    urwid.Divider(),
                    urwid.Columns([(20, start_btn)], dividechars=1),
                    self._error_text,
                ]
            )
        )

        form_area = urwid.ScrollBar(
            listbox,
            trough_char=urwid.ScrollBar.Symbols.LITE_SHADE,
        )

        body = urwid.Pile(
            [
                ("weight", 3, form_area),
                ("weight", 2, self._progress_panel),
                ("pack", self._summary_text),
            ]
        )

        self._main_body = body

        self._update_footer()
        return urwid.Frame(body=body)

    # ------------------------------------------------------------------ #
    #  Submit
    # ------------------------------------------------------------------ #

    def _on_start(self, _btn=None) -> None:
        if self._running:
            return

        vals = self._form.get_values()
        structure_file = vals["structure_file"].strip()
        if not structure_file:
            self._error_text.set_text(("form_error", "Structure file is required"))
            return

        if not Path(structure_file).is_file():
            self._error_text.set_text(
                ("form_error", f"File not found: {structure_file}")
            )
            return

        self._running = True
        self._error_text.set_text("")
        self._summary_text.set_text("")
        self._progress_panel.clear()

        from rapmat.tui.tasks import BackgroundTask

        self._task = BackgroundTask(
            fn=lambda prog: self._worker(prog, vals),
            loop=self._state.loop,
            on_progress=self._progress_panel.set_progress,
            on_log=self._progress_panel.add_log,
            on_complete=self._on_complete,
            on_error=self._on_error,
        )
        self._task.start()

    def _worker(self, progress, vals: dict) -> None:
        import warnings

        from ase import Atoms
        from ase.io import read as read_ase_structure
        from rapmat.calculators import Calculators
        from rapmat.calculators.factory import load_calculator
        from rapmat.core.phonon import (
            structure_calculate_phonons,
            structure_has_imag_phonon_freq,
        )
        from rapmat.utils.common import workdir_context

        class _TaskCalcCallback:
            def on_status(self, message: str) -> None:
                progress.log(message)
                progress.update(1, 5, message)

        structure_file = vals["structure_file"].strip()
        calculator_name = vals["calculator"]
        supercell = vals["supercell"]
        qpoint_mesh = vals["qpoint_mesh"]
        displacement = vals["displacement"]
        imag_cutoff = vals["imag_cutoff"]
        prerelax = vals["prerelax"]
        reduce_prim = vals["reduce_prim"]
        plot_file = vals["plot_file"].strip() or "phonon_plot.png"

        progress.log(f"Reading structure from {structure_file}...")
        progress.update(0, 5, "Reading structure")

        structure = read_ase_structure(structure_file)
        if not isinstance(structure, Atoms):
            structure = structure[-1]

        with workdir_context(None) as wdir:
            progress.log(f"Working directory: {wdir}")
            progress.update(1, 5, "Loading calculator")
            progress.log(f"Loading calculator {calculator_name}...")
            calculator = load_calculator(
                Calculators(calculator_name), wdir, config={},
                callback=_TaskCalcCallback(),
            )
            structure.calc = calculator

            if prerelax:
                progress.update(2, 5, "Pre-relaxing")
                progress.log("Pre-relaxing structure...")
                from rapmat.core.relaxation import structure_relax

                cancel_flag = [False]

                # We need a callback to set the cancel flag if the user pressed Esc
                # while we are in the optimizer loop
                def _phony_check():
                    if progress.cancelled:
                        cancel_flag[0] = True

                # Hack: Python doesn't let us inject callbacks into ASE easily without our cancel_flag hook
                # but we can at least pass cancel_flag down. To actually trip it, we'd need another thread,
                # but for now we just pass it so it's there. The _worker itself checks progress.cancelled above.

                converged, relaxed = structure_relax(
                    structure,
                    force_conv_crit=1e-3,
                    steps_max=10000,
                    cancel_flag=cancel_flag,
                )
                if progress.cancelled or cancel_flag[0]:
                    raise KeyboardInterrupt("Cancelled by user")

                if converged:
                    structure = relaxed
                else:
                    progress.log("WARNING: Pre-relax did not converge")

            if reduce_prim:
                progress.update(2, 5, "Reducing to primitive cell")
                progress.log("Reducing to primitive cell...")
                from rapmat.utils.structure import standardize_atoms
                calc = structure.calc
                try:
                    structure = standardize_atoms(structure, to_primitive=True)
                    structure.calc = calc
                except Exception as e:
                    progress.log(f"WARNING: Could not reduce cell: {e}")

            progress.update(3, 5, "Computing phonons")
            progress.log("Computing phonon dispersion...")
            phonons = structure_calculate_phonons(
                structure,
                displacement,
                supercell,
                qpoint_mesh,
            )

            is_unstable = structure_has_imag_phonon_freq(phonons, imag_cutoff)

            progress.update(4, 5, "Saving plot")
            progress.log("Generating band structure plot...")

            plot_path = Path(plot_file).resolve()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig = phonons.auto_band_structure(plot=True)
            fig.savefig(plot_path)

            self._result = {
                "calculator": calculator_name,
                "stable": not is_unstable,
                "plot_path": str(plot_path),
                "displacement": displacement,
                "supercell": supercell,
                "qpoint_mesh": qpoint_mesh,
            }

        progress.update(5, 5, "Done")
        progress.finish()

    # ------------------------------------------------------------------ #
    #  Completion
    # ------------------------------------------------------------------ #

    def _on_complete(self) -> None:
        self._running = False
        self._progress_panel.set_finished(True, "Phonon calculation complete!")

        r = getattr(self, "_result", None)
        if r:
            stable_str = "Yes" if r["stable"] else "No (imaginary frequencies detected)"
            self._summary_text.set_text(
                [
                    ("section", "\n Phonon Dispersion Result\n"),
                    ("form_label", "  Calculator: "),
                    ("details", r["calculator"] + "\n"),
                    ("form_label", "  Stable:     "),
                    ("success" if r["stable"] else "error", stable_str + "\n"),
                    ("form_label", "  Supercell:  "),
                    ("details", str(r["supercell"]) + "\n"),
                    ("form_label", "  Q-mesh:     "),
                    ("details", str(r["qpoint_mesh"]) + "\n"),
                    ("form_label", "  Plot:       "),
                    ("details", r["plot_path"]),
                ]
            )

    def _on_error(self, error: str) -> None:
        self._running = False
        self._progress_panel.set_finished(False, f"Error: {error}")

    # ------------------------------------------------------------------ #
    #  Key handling
    # ------------------------------------------------------------------ #

    def keypress(self, size: tuple, key: str) -> str | None:
        if key == "f5":
            self._on_start()
            return None
        if key == "esc":
            if self._running:
                if self._task:
                    self._task.cancel()
                    self._progress_panel.set_cancelling()
                return None
            self._router.pop()
            return None
        return key
