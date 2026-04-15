import uuid
import urwid

from rapmat.tui.widgets.dialog import ModalDialog
from rapmat.tui.widgets.progress import ProgressPanel
from rapmat.tui.widgets.form import (
    FormGroup,
    checkbox_field,
    dropdown_field,
    float_field,
    int_field,
    radio_field,
    text_field,
)


from rapmat.tui.router import ScreenRouter
from rapmat.tui.state import AppState
from rapmat.tui.tasks import BackgroundTask


def _calc_options() -> list[str]:
    from rapmat.calculators import Calculators

    return [c.value for c in Calculators]


class CSPSearchScreen:
    title = "New CSP Run"

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        self._state = state
        self._router = router
        self._frame: urwid.Frame | None = None
        self._main_body: urwid.Widget | None = None
        self._progress_panel = ProgressPanel(title=" Run Progress ")
        self._task: BackgroundTask | None = None
        self._running = False

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
        if self._task is not None:
            self._task.cancel()

    def _update_footer(self) -> None:
        if self._state.status_bar:
            self._state.status_bar.set_keys(
                [
                    ("F5", "Start Run"),
                    ("Esc", "Back"),
                ]
            )
            self._state.status_bar.clear_message()

    # ------------------------------------------------------------------ #
    #  Form construction
    # ------------------------------------------------------------------ #

    def _study_options(self) -> list[str]:
        if not self._state.studies_cache:
            return ["— (no studies available)"]
        names = []
        for s in self._state.studies_cache:
            sid = s.get("study_id") or s.get("id") or s.get("name", "")
            if isinstance(sid, str) and ":" in sid:
                sid = sid.split(":")[-1]
            names.append(str(sid))
        return names

    def _build_form(self) -> FormGroup:
        options = self._study_options()
        default_idx = 0
        if self._state.active_study and self._state.active_study in options:
            default_idx = options.index(self._state.active_study)

        return FormGroup(
            [
                dropdown_field("study", "Study", options, default=default_idx),
                text_field(
                    "formula",
                    "Formula (e.g. Al2O3)",
                    default="",
                    validator=self._validate_formula,
                ),
                checkbox_field(
                    "grid_search", "Compositional Grid Search", default=False
                ),
                float_field("grid_step", "Grid Step Size", default=0.25),
                int_field("fu_min", "Formula units min", default=2),
                int_field("fu_max", "Formula units max", default=4),
                int_field("candidates", "Candidates/group", default=2),
                text_field("run_name", "Run name", default=""),
                int_field("seed", "Seed (0 = auto)", default=0),
                int_field("workers", "Workers (CPU)", default=1),
            ],
            label_width=26,
        )

    @staticmethod
    def _validate_formula(val):
        if not val or not val.strip():
            return "Formula is required"
        try:
            from rapmat.utils.common import parse_formula

            parts = parse_formula(val.strip())
            if len(parts) > 2:
                pass
        except Exception:
            return "Invalid formula (e.g. Al2O3 or Al-O for grid search)"
        return None

    # ------------------------------------------------------------------ #
    #  Layout
    # ------------------------------------------------------------------ #

    def _sync_grid_formula(self, _widget=None, _state=None) -> None:
        grid_cb = self._form.get_widget("grid_search")
        study_dd = self._form.get_widget("study")

        if _widget == grid_cb:
            is_grid = _state
            study_raw = study_dd.value
        elif _widget == study_dd:
            is_grid = grid_cb.get_state()
            study_raw = _state
        else:
            vals = self._form.get_values()
            is_grid = vals.get("grid_search", False)
            study_raw = vals.get("study", "— (none)")

        has_study = not (
            is_grid is False
            and isinstance(study_raw, str)
            and study_raw.startswith("—")
        )
        if isinstance(study_raw, str) and study_raw.startswith("—"):
            has_study = False
        else:
            has_study = True

        disable_formula = is_grid and has_study

        self._form.set_field_disabled("formula", disable_formula)

        if disable_formula:
            study = self._state.store.get_study(study_raw)
            if study and study.get("system"):
                self._form.set_values({"formula": study["system"]})

    def _build_frame(self) -> urwid.Frame:
        self._form = self._build_form()

        study_dd = self._form.get_widget("study")
        urwid.connect_signal(study_dd, "change", self._sync_grid_formula)
        grid_cb = self._form.get_widget("grid_search")
        urwid.connect_signal(grid_cb, "change", self._sync_grid_formula)

        self._error_text = urwid.Text("")

        listbox = urwid.ListBox(
            urwid.SimpleListWalker(
                [
                    self._form,
                    urwid.Divider(),
                    self._error_text,
                ]
            )
        )

        form_box = urwid.ScrollBar(
            listbox,
            trough_char=urwid.ScrollBar.Symbols.LITE_SHADE,
        )

        start_btn = urwid.AttrMap(
            urwid.Button("Start Run [F5]", on_press=self._on_start),
            "menu_item",
            focus_map="btn_focus",
        )

        body = urwid.Pile(
            [
                ("weight", 3, form_box),
                ("pack", urwid.Columns([(20, start_btn)], dividechars=1)),
                ("pack", urwid.Divider()),
                ("weight", 2, self._progress_panel),
            ]
        )

        self._main_body = body

        self._update_footer()
        return urwid.Frame(body=body)

    # ------------------------------------------------------------------ #
    #  Submit handler
    # ------------------------------------------------------------------ #

    def _on_start(self, _btn=None) -> None:
        if self._running:
            return

        errs = self._form.validate()
        if errs:
            self._error_text.set_text(("form_error", "\n".join(errs)))
            return

        self._error_text.set_text("")
        vals = self._form.get_values()
        self._running = True
        self._progress_panel.clear()
        self._progress_panel.add_log("Preparing run...")

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

    def _worker(self, progress, vals) -> None:
        import time
        from rapmat.core.csp import run_generation_loop, run_processing_loop
        from rapmat.utils.common import parse_formula, workdir_context

        store = self._state.store
        fu_min = max(1, vals["fu_min"])
        fu_max = max(fu_min, vals["fu_max"])
        candidates = max(1, vals["candidates"])
        base_run_name = vals["run_name"].strip() or f"run-{uuid.uuid4().hex[:8]}"
        workers = max(1, vals["workers"])

        is_grid_search = vals["grid_search"]
        grid_step = vals["grid_step"]

        formula_str = vals.get("formula", "")
        formula = parse_formula(formula_str) if formula_str else {}
        elements = list(formula.keys())

        import random as _random

        seed_val = vals.get("seed", 0)
        if seed_val == 0:
            seed_val = _random.randint(1, 2**32 - 1)
        progress.log(f"Using seed: {seed_val}")

        study_raw = vals["study"]
        if study_raw.startswith("—"):
            raise ValueError("You must select a valid Study to start a run.")

        study_id = study_raw

        study = store.get_study(study_id)
        if not study:
            raise ValueError(f"Study {study_id} no longer exists.")

        domain = study.get("domain", "bulk")
        search_dim = 3 if domain == "bulk" else 2

        from rapmat.storage import SOAPDescriptor

        descriptor = SOAPDescriptor(species=elements)
        vec_col = store.register_descriptor(
            descriptor.descriptor_id(),
            descriptor.dimension(),
            meta={"type": "SOAP", "species": elements},
        )

        runs_to_queue = []
        if is_grid_search:
            el_A, el_B = elements[0], elements[1]
            import numpy as np

            ratios = np.arange(0.0, 1.0 + grid_step / 2, grid_step)
            for x in ratios:
                y = 1.0 - x

                import math

                def float_to_ratio(val, tol=1e-4):
                    for i in range(1, 100):
                        if abs(val * i - round(val * i)) < tol:
                            return round(val * i), i
                    return round(val * 100), 100

                nx, dx = float_to_ratio(x)
                ny, dy = float_to_ratio(y)
                lcm = abs(dx * dy) // math.gcd(dx, dy)

                int_x = int(nx * (lcm / dx))
                int_y = int(ny * (lcm / dy))

                # Simplify
                common = math.gcd(int_x, int_y)
                if common > 0:
                    int_x //= common
                    int_y //= common

                sub_formula = {}
                label = ""
                if int_x > 0:
                    sub_formula[el_A] = int_x
                    label += f"{el_A}{int_x if int_x > 1 else ''}"
                if int_y > 0:
                    sub_formula[el_B] = int_y
                    label += f"{el_B}{int_y if int_y > 1 else ''}"

                sub_run_name = f"{base_run_name}-{label}"
                runs_to_queue.append((sub_run_name, sub_formula))
        else:
            runs_to_queue.append((base_run_name, formula))

        wid = uuid.uuid4().hex[:12]

        def _cb(current, total, msg, is_log=True):
            if progress.cancelled:
                raise KeyboardInterrupt("Cancelled by user")
            progress.update(current, total, msg)
            if is_log:
                progress.log(msg)

        total_runs = len(runs_to_queue)
        runs_created = []

        for i, (run_name, run_formula) in enumerate(runs_to_queue):
            progress.log(f"[{i+1}/{total_runs}] Queuing run '{run_name}'...")

            run_config = {
                "formula": run_formula,
                "formula_units": [fu_min, fu_max],
                "candidates_per_group": candidates,
                "skip_not_converged": False,
                "descriptor_id": descriptor.descriptor_id()[:12],
                "vec_col": vec_col,
                "seed": seed_val,
            }

            try:
                store.create_run(
                    name=run_name,
                    study_id=study_id,
                    config=run_config,
                    worker_id=wid,
                )
            except Exception as e:
                progress.log(f"Failed to create run {run_name}: {e}")
                continue

            spg_total = 230 if search_dim == 3 else 80
            placeholders = []
            idx = 0
            for fu in range(fu_min, fu_max + 1):
                for spg in range(1, spg_total + 1):
                    for _ in range(candidates):
                        idx += 1
                        placeholders.append((f"{run_name}/{idx}", spg, fu))

            store.add_generation_placeholders(run_name, placeholders)
            runs_created.append((run_name, run_config))

        for i, (run_name, run_config) in enumerate(runs_created):
            cancel_flag = [False]

            progress.log(
                f"[{i+1}/{len(runs_created)}] Starting generation phase for {run_name}..."
            )
            try:
                with workdir_context(None) as workdir_path:
                    progress.log(f"Working directory: {workdir_path}")

                    meta = store.get_run_metadata(run_name) or {}
                    full_cfg = meta.get("config", run_config)

                    run_generation_loop(
                        run_name=run_name,
                        store=store,
                        config=full_cfg,
                        worker_id=wid,
                        descriptor=descriptor,
                        workers=workers,
                        progress_callback=_cb,
                        cancel_flag=cancel_flag,
                        log_callback=progress.log,
                    )

                    if progress.cancelled or cancel_flag[0]:
                        raise KeyboardInterrupt("Cancelled by user")

                    progress.log(
                        "Generation complete. Initializing calculator for processing..."
                    )
                    store.set_run_status(run_name, "processing")

                    def _proc_cb(current, total, msg, is_log=True):
                        if progress.cancelled:
                            cancel_flag[0] = True
                            raise KeyboardInterrupt("Cancelled by user")
                        progress.update(current, total, msg)
                        if is_log:
                            progress.log(msg)

                    progress.log(
                        f"[{i+1}/{len(runs_created)}] Starting processing phase for {run_name}..."
                    )

                    t0 = time.monotonic()
                    run_processing_loop(
                        run_name=run_name,
                        store=store,
                        config=full_cfg,
                        workdir_path=workdir_path,
                        descriptor=descriptor,
                        worker_id=wid,
                        progress_callback=_proc_cb,
                        cancel_flag=cancel_flag,
                    )
                    t1 = time.monotonic()
                    progress.log(
                        f"Run '{run_name}' computation finished in {t1 - t0:.2f} seconds."
                    )

                    store.release_run(run_name, "completed")
            except KeyboardInterrupt:
                store.release_run(run_name, "interrupted")
                raise
            except Exception:
                store.release_run(run_name, "failed")
                raise

        if runs_to_queue:
            self._state.active_run = runs_to_queue[-1][0]

        self._state.invalidate()
        progress.finish()

    # ------------------------------------------------------------------ #
    #  Completion callbacks
    # ------------------------------------------------------------------ #

    def _on_complete(self) -> None:
        self._running = False
        self._progress_panel.set_finished(True, "Run completed successfully!")
        if self._frame and self._main_body:
            dlg = ModalDialog.confirm(
                "Run Complete",
                "CSP run finished. View results?",
                parent=self._main_body,
                on_close=self._on_dialog_close,
            )
            self._frame.body = dlg

    def _on_error(self, error: str) -> None:
        self._running = False
        self._progress_panel.set_finished(False, f"Error: {error}")
        self._progress_panel.add_log(f"ERROR: {error}")

    def _on_dialog_close(self, confirmed: bool) -> None:
        if self._frame:
            self._frame.body = self._main_body
        if confirmed and self._state.active_run:
            from rapmat.tui.screens.results import ResultsScreen

            self._router.push(ResultsScreen(self._state, self._router))

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
