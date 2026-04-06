"""Evaluation screen for the Rapmat TUI."""

import json
from typing import List

import urwid

from rapmat.tui.widgets.form import (
    FormGroup,
    checkbox_field,
    dropdown_field,
    float_field,
    int_field,
    text_field,
    tuple_field,
)
from rapmat.tui.widgets.progress import ProgressPanel
from rapmat.tui.widgets.table import SortableTable

from rapmat.tui.router import ScreenRouter
from rapmat.tui.state import AppState
from rapmat.tui.tasks import BackgroundTask


def _calc_options() -> list[str]:
    from rapmat.calculators import Calculators

    return [c.value for c in Calculators]


_RESULT_COLS = [
    ("ID", 22),
    ("Formula", 10),
    ("MLIP eV/at", 12),
    ("Ref eV/at", 12),
    ("Diff", 10),
    ("MLIP Dyn", 10),
    ("Ref Dyn", 10),
]


class EvalScreen:
    """Run evaluation against a reference calculator."""

    title = "Evaluation"

    @property
    def breadcrumb_title(self) -> str:
        run = self._state.active_run
        return f"Evaluation: {run}" if run else self.title

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        self._state = state
        self._router = router
        self._frame: urwid.Frame | None = None
        self._main_body: urwid.Widget | None = None
        self._progress_panel = ProgressPanel(title=" Evaluation Progress ")
        self._task: BackgroundTask | None = None
        self._running = False
        self._comparison: List[dict] = []
        self._records: list[dict] = []

    # ------------------------------------------------------------------ #
    #  Screen protocol
    # ------------------------------------------------------------------ #

    def build(self) -> urwid.Widget:
        self._state.refresh_runs_if_needed()
        self._frame = self._build_frame()
        return self._frame

    def on_resume(self) -> None:
        self._update_footer()

    def _update_footer(self) -> None:
        if self._state.status_bar:
            self._state.status_bar.set_keys(
                [
                    ("F5", "Evaluate"),
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

    def _run_options(self) -> list[str]:
        names = [r.get("name", "") for r in self._state.runs_cache]
        if self._state.active_run and self._state.active_run not in names:
            names.insert(0, self._state.active_run)
        return names if names else ["(no runs)"]

    def _build_frame(self) -> urwid.Frame:
        run_opts = self._run_options()
        default_idx = 0
        if self._state.active_run and self._state.active_run in run_opts:
            default_idx = run_opts.index(self._state.active_run)

        self._form = FormGroup(
            [
                dropdown_field("run_name", "Run", run_opts, default=default_idx),
                dropdown_field(
                    "calculator", "Ref calculator", _calc_options(), default=0
                ),
                text_field(
                    "calculator_config", "Config TOML Path", default=""
                ),
                int_field("top_n", "Top N (0=all)", default=0),
                checkbox_field("run_phonons", "Run phonons", default=False),
                checkbox_field("stable_only", "Dyn. stable only (tau)", default=False),
                tuple_field(
                    "phonon_supercell", "Phonon supercell", size=3, default=(3, 3, 3)
                ),
                tuple_field("phonon_mesh", "Phonon mesh", size=3, default=(20, 20, 20)),
                float_field("phonon_displacement", "Phonon displacement", default=1e-2),
                float_field("phonon_cutoff", "Phonon cutoff", default=-0.15),
            ],
            label_width=22,
        )

        self._error_text = urwid.Text("")
        self._results_pile = urwid.Pile([])

        start_btn = urwid.AttrMap(
            urwid.Button("Evaluate [F5]", on_press=self._on_start),
            "menu_item",
            focus_map="btn_focus",
        )

        listbox_form = urwid.ListBox(
            urwid.SimpleListWalker(
                [
                    self._form,
                    urwid.Divider(),
                    urwid.Columns([(18, start_btn)], dividechars=1),
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
                ("weight", 1, self._progress_panel),
                ("weight", 2, self._results_pile),
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

        self._running = True
        self._error_text.set_text("")
        self._progress_panel.clear()
        self._results_pile.contents[:] = []

        vals = self._form.get_values()
        
        calc_config_dict = {}
        calc_config_path = vals.get("calculator_config", "").strip()
        if calc_config_path:
            import tomllib
            from pathlib import Path

            config_file = Path(calc_config_path)
            if not config_file.is_file():
                self._error_text.set_text(
                    ("form_error", f"Config file not found: {calc_config_path}")
                )
                self._running = False
                return
            try:
                with open(config_file, "rb") as f:
                    calc_config_dict = tomllib.load(f)
            except Exception as e:
                self._error_text.set_text(
                    ("form_error", f"Invalid TOML in config: {e}")
                )
                self._running = False
                return

        vals["calculator_config_dict"] = calc_config_dict

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
        from rapmat.calculators import Calculators
        from rapmat.calculators.factory import load_calculator
        from rapmat.core.evaluation import run_eval_loop
        from rapmat.utils.common import workdir_context

        class _TaskCalcCallback:
            def on_status(self, message: str) -> None:
                progress.log(message)

        store = self._state.store
        run_name = vals["run_name"]
        calculator_name = vals["calculator"]
        top_n = vals["top_n"]
        run_phonons = vals["run_phonons"]

        progress.log(f"Loading structures for '{run_name}'...")
        records = store.get_run_structures(run_name, status="relaxed")
        records.sort(key=lambda r: r["energy_per_atom"])

        if not records:
            progress.fail("No relaxed structures found")
            return

        if top_n and top_n > 0:
            records = records[:top_n]

        self._records = records
        config_dict = {"run_phonons": run_phonons}
        config_dict["calculator_config"] = vals.get("calculator_config_dict", {})
        if run_phonons:
            config_dict["phonon_supercell"] = vals.get("phonon_supercell", (3, 3, 3))
            config_dict["phonon_mesh"] = vals.get("phonon_mesh", (20, 20, 20))
            config_dict["phonon_displacement"] = vals.get("phonon_displacement", 1e-2)
            
        config_json = json.dumps(config_dict, sort_keys=True)

        pending = [
            r
            for r in records
            if not store.has_evaluation(r["id"], calculator_name, config_json)
        ]

        progress.log(f"{len(records)} structures, {len(pending)} to evaluate")

        if pending:
            with workdir_context(None) as wdir:
                progress.log(f"Working directory: {wdir}")
                calculator = load_calculator(
                    Calculators(calculator_name),
                    wdir,
                    config=vals.get("calculator_config_dict", {}),
                    callback=_TaskCalcCallback(),
                )

                def _cb(current, total, msg, is_log=True):
                    if progress.cancelled:
                        raise KeyboardInterrupt("Cancelled")
                    progress.update(current, total, msg)
                    if is_log:
                        progress.log(msg)

                run_eval_loop(
                    pending,
                    store,
                    run_name,
                    calculator,
                    calculator_name,
                    config_json,
                    run_phonons=run_phonons,
                    phonon_displacement=vals.get("phonon_displacement", 1e-2),
                    phonon_supercell=vals.get("phonon_supercell", (3, 3, 3)),
                    phonon_mesh=vals.get("phonon_mesh", (20, 20, 20)),
                    progress_callback=_cb,
                    log_callback=progress.log,
                )

        evals = store.get_evaluations(run_name, calculator=calculator_name)
        eval_map = {ev["structure_id"]: ev for ev in evals if ev.get("config_json") == config_json}

        comparison = []
        for rec in records:
            ev = eval_map.get(rec["id"])
            if ev is None:
                continue
            comparison.append(
                {
                    "id": rec["id"],
                    "formula": rec.get("formula", ""),
                    "mlip_epa": rec["energy_per_atom"],
                    "ref_epa": ev["energy_per_atom"],
                    "mlip_phonon_freq": rec.get("min_phonon_freq"),
                    "ref_phonon_freq": ev.get("min_phonon_freq"),
                }
            )

        self._comparison = comparison
        progress.finish()

    # ------------------------------------------------------------------ #
    #  Completion
    # ------------------------------------------------------------------ #

    def _on_complete(self) -> None:
        self._running = False
        self._progress_panel.set_finished(True, "Evaluation complete!")

        if not self._comparison:
            self._error_text.set_text(("unconv", "No evaluation results."))
            return

        vals = self._form.get_values()
        phonon_cutoff = vals.get("phonon_cutoff", -0.15)
        stable_only = vals.get("stable_only", True)

        try:
            from rapmat.core.evaluation import (
                compute_ranking_metrics,
                compute_stability_metrics,
            )

            ranking = compute_ranking_metrics(
                self._comparison, phonon_cutoff, stable_only
            )
            stability = compute_stability_metrics(self._comparison, phonon_cutoff)
        except Exception as exc:
            ranking = {}
            stability = None
            if self._error_text is not None:
                self._error_text.set_text(
                    ("form_error", f"  Metrics computation failed: {exc}")
                )

        lines = []
        if ranking.get("kendall_tau") is not None:
            lines.append(
                f"Kendall tau: {ranking['kendall_tau']:.4f}  "
                f"(p={ranking['p_value']:.2e}, n={ranking['n_structures']})"
            )
        if ranking.get("mae_epa") is not None:
            lines.append(f"MAE eV/atom: {ranking['mae_epa']:.4f}")
        if stability is not None:
            lines.append(
                f"Dyn. stability F1: {stability['f1']:.4f}  "
                f"(P={stability['precision']:.2f}, R={stability['recall']:.2f})"
            )

        summary = urwid.Text(
            ("details", "\n".join(lines) if lines else "No metrics available")
        )

        def _fmt_dyn(val: float | None) -> str:
            if val is None:
                return "N/A"
            return "Yes" if val >= phonon_cutoff else "No"

        table = SortableTable(
            columns=_RESULT_COLS,
            row_data=self._comparison,
            format_row=lambda c: [
                c["id"][:20],
                c.get("formula", ""),
                f"{c['mlip_epa']:.4f}",
                f"{c['ref_epa']:.4f}",
                f"{c['ref_epa'] - c['mlip_epa']:+.4f}",
                _fmt_dyn(c.get("mlip_phonon_freq")),
                _fmt_dyn(c.get("ref_phonon_freq")),
            ],
        )

        self._results_pile.contents[:] = [
            (
                urwid.Text(
                    ("section", f" Results ({len(self._comparison)} structures)")
                ),
                ("pack", None),
            ),
            (urwid.Divider("─"), ("pack", None)),
            (summary, ("pack", None)),
            (urwid.Divider(), ("pack", None)),
            (table, ("weight", 1)),
        ]

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
