"""Dedup analysis screen for the Rapmat TUI."""

from pathlib import Path


import urwid

from rapmat.tui.widgets.form import (
    FormGroup,
    checkbox_field,
    dropdown_field,
    float_field,
)
from rapmat.tui.widgets.progress import ProgressPanel
from rapmat.tui.widgets.table import SortableTable

from rapmat.tui.router import ScreenRouter
from rapmat.tui.state import AppState
from rapmat.tui.tasks import BackgroundTask

_DIST_COLS = [("Metric", 28), ("Value", 18)]
_WATERFALL_COLS = [("Stage", 24), ("Kept", 10), ("Change", 10), ("Notes", 30)]
_COLLISION_COLS = [
    ("Check", 24),
    ("Comparisons", 14),
    ("Agreements", 12),
    ("Disagreements", 14),
    ("Rate", 8),
]


class DedupScreen:
    """Deduplication analysis screen."""

    title = "Dedup Analysis"

    @property
    def breadcrumb_title(self) -> str:
        run = self._state.active_run
        return f"Dedup: {run}" if run else self.title

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        self._state = state
        self._router = router
        self._frame: urwid.Frame | None = None
        self._main_body: urwid.Widget | None = None
        self._progress_panel = ProgressPanel(title=" Dedup Progress ")
        self._task: BackgroundTask | None = None
        self._running = False
        self._result_data: dict | None = None

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
                    ("F5", "Analyze"),
                    ("p", "Save plot"),
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
                dropdown_field("stage", "Stage", ["relaxed", "candidates"], default=0),
                float_field("dedup_threshold", "Dedup threshold", default=1e-2),
                checkbox_field("pymatgen_dedup", "Pymatgen dedup", default=False),
                float_field("pymatgen_ltol", "Pymatgen ltol", default=0.2),
                float_field("pymatgen_stol", "Pymatgen stol", default=0.3),
                float_field("pymatgen_angle", "Pymatgen angle tol", default=5.0),
                checkbox_field("force_dedup", "Force dedup", default=False),
                float_field("force_cosine", "Force cosine thresh", default=0.95),
            ],
            label_width=22,
        )

        self._error_text = urwid.Text("")
        self._results_pile = urwid.Pile([])

        start_btn = urwid.AttrMap(
            urwid.Button("Analyze [F5]", on_press=self._on_start),
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

        listbox_results = urwid.ListBox(urwid.SimpleListWalker([self._results_pile]))
        results_area = urwid.ScrollBar(
            listbox_results,
            trough_char=urwid.ScrollBar.Symbols.LITE_SHADE,
        )

        body = urwid.Pile(
            [
                ("weight", 2, form_area),
                ("weight", 1, self._progress_panel),
                ("weight", 3, results_area),
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
        import numpy as np
        from rapmat.core.dedup_analysis import (
            compute_pairwise_distances,
            find_threshold_for_survival,
            simulate_deduplication,
        )

        store = self._state.store
        run_name = vals["run_name"]
        stage = vals["stage"]
        threshold = vals["dedup_threshold"]

        meta = store.get_run_metadata(run_name)
        if meta is None:
            progress.fail(f"Run '{run_name}' not found")
            return

        run_config = meta.get("config", {})
        _formula = run_config.get("formula", {})
        _elements = list(_formula.keys()) if isinstance(_formula, dict) else []
        if not _elements:
            progress.fail("Cannot determine species from run config")
            return

        from rapmat.storage import SOAPDescriptor

        descriptor = SOAPDescriptor(species=_elements)

        statuses = ("generated",) if stage == "candidates" else ("relaxed",)

        progress.log(f"Loading {stage} structures...")
        progress.update(0, 5, "Loading structures")
        structures = store.get_structures_for_analysis(run_name, statuses=statuses)

        if not structures:
            progress.fail(f"No {stage} structures found")
            return

        if len(structures) < 2:
            progress.fail("Need >= 2 structures for analysis")
            return

        progress.log(f"Loaded {len(structures)} structures, computing vectors...")
        progress.update(1, 5, "Computing descriptor vectors")
        for s in structures:
            s["vector"] = descriptor.compute(s["atoms"])

        progress.update(2, 5, "Computing pairwise distances")

        vectors = np.vstack([s["vector"] for s in structures])
        distances = compute_pairwise_distances(vectors)

        progress.update(3, 5, "Simulating deduplication")
        progress.log("Simulating dedup pipeline...")

        def _dedup_cb(current, total, is_log=False):
            msg = f"Dedup: {current}/{total}"
            progress.update(current, total, msg)
            if is_log:
                progress.log(msg)

        sim = simulate_deduplication(
            structures,
            threshold=threshold,
            use_pymatgen=vals["pymatgen_dedup"],
            ltol=vals["pymatgen_ltol"],
            stol=vals["pymatgen_stol"],
            angle_tol=vals["pymatgen_angle"],
            use_forces=vals["force_dedup"],
            force_cosine_threshold=vals["force_cosine"],
            progress_callback=_dedup_cb,
        )

        progress.update(4, 5, "Computing percentiles")
        max_dist = float(np.max(distances)) if len(distances) > 0 else 1.0
        targets = [95, 90, 75, 50, 25, 10, 5]
        percentiles = []
        for p in targets:
            thresh, kept = find_threshold_for_survival(structures, p / 100.0, max_dist)
            percentiles.append((p, thresh, kept))

        n_pairs = len(distances)
        below = int(np.sum(distances < threshold))

        self._result_data = {
            "n_structs": len(structures),
            "n_pairs": n_pairs,
            "min_dist": float(np.min(distances)),
            "max_dist": float(np.max(distances)),
            "mean_dist": float(np.mean(distances)),
            "median_dist": float(np.median(distances)),
            "std_dist": float(np.std(distances)),
            "below_thresh": below,
            "threshold": threshold,
            "sim": sim,
            "percentiles": percentiles,
            "distances": distances,
            "stage": stage,
            "run_name": run_name,
            "use_pymatgen": vals["pymatgen_dedup"],
            "use_forces": vals["force_dedup"],
        }

        progress.update(4, 4, "Done")
        progress.finish()

    # ------------------------------------------------------------------ #
    #  Completion
    # ------------------------------------------------------------------ #

    def _on_complete(self) -> None:
        self._running = False
        self._progress_panel.set_finished(True, "Analysis complete!")

        d = self._result_data
        if d is None:
            return
        sim = d["sim"]

        dist_rows = [
            {"metric": "Structures (with vectors)", "value": str(d["n_structs"])},
            {"metric": "Total pairs", "value": str(d["n_pairs"])},
            {"metric": "Min distance", "value": f"{d['min_dist']:.6f}"},
            {"metric": "Max distance", "value": f"{d['max_dist']:.6f}"},
            {"metric": "Mean distance", "value": f"{d['mean_dist']:.6f}"},
            {"metric": "Median distance", "value": f"{d['median_dist']:.6f}"},
            {"metric": "Std deviation", "value": f"{d['std_dist']:.6f}"},
            {
                "metric": f"Pairs below {d['threshold']}",
                "value": f"{d['below_thresh']} ({100*d['below_thresh']/max(d['n_pairs'],1):.1f}%)",
            },
        ]
        dist_table = SortableTable(
            columns=_DIST_COLS,
            row_data=dist_rows,
            format_row=lambda r: [r["metric"], r["value"]],
        )

        waterfall_rows = [
            {
                "stage": "Initial",
                "kept": str(sim.total),
                "change": "",
                "notes": f"All {d['stage']}",
            },
            {
                "stage": "Stage 1: Vector (L2)",
                "kept": str(
                    sim.total
                    - sim.dropped_by_vector
                    - sim.rescued_by_pymatgen
                    - sim.rescued_by_forces
                ),
                "change": f"-{sim.dropped_by_vector + sim.rescued_by_pymatgen + sim.rescued_by_forces}",
                "notes": f"threshold < {d['threshold']}",
            },
        ]
        if d["use_pymatgen"]:
            after_vec = (
                sim.total
                - sim.dropped_by_vector
                - sim.rescued_by_pymatgen
                - sim.rescued_by_forces
            )
            waterfall_rows.append(
                {
                    "stage": "Stage 2: Pymatgen",
                    "kept": str(after_vec + sim.rescued_by_pymatgen),
                    "change": (
                        f"+{sim.rescued_by_pymatgen}"
                        if sim.rescued_by_pymatgen
                        else "0"
                    ),
                    "notes": f"{sim.pymatgen_mismatches}/{sim.pymatgen_comparisons} collisions",
                }
            )
        if d["use_forces"]:
            waterfall_rows.append(
                {
                    "stage": "Stage 3: Forces",
                    "kept": str(sim.kept),
                    "change": (
                        f"+{sim.rescued_by_forces}" if sim.rescued_by_forces else "0"
                    ),
                    "notes": f"{sim.force_mismatches}/{sim.force_comparisons} disagreements",
                }
            )
        waterfall_rows.append(
            {
                "stage": "Final",
                "kept": str(sim.kept),
                "change": "",
                "notes": f"{sim.final_dropped} dropped, {sim.kept} unique",
            }
        )

        waterfall_table = SortableTable(
            columns=_WATERFALL_COLS,
            row_data=waterfall_rows,
            format_row=lambda r: [r["stage"], r["kept"], r["change"], r["notes"]],
        )

        # Percentile table
        perc_text_lines = ["Survival Percentile Thresholds:"]
        for p, thresh, kept in d["percentiles"]:
            perc_text_lines.append(
                f"  {p:3d}%  threshold={thresh:.4f}  kept={kept}/{sim.total}"
            )

        widgets: list[tuple] = [
            (urwid.Text(("section", " Distance Statistics")), ("pack", None)),
            (urwid.Divider("─"), ("pack", None)),
            (dist_table, ("given", 12)),
            (urwid.Divider(), ("pack", None)),
            (urwid.Text(("section", " Dedup Simulation Waterfall")), ("pack", None)),
            (urwid.Divider("─"), ("pack", None)),
            (waterfall_table, ("given", 8)),
            (urwid.Divider(), ("pack", None)),
            (urwid.Text(("details", "\n".join(perc_text_lines))), ("pack", None)),
        ]

        # Collision table
        if d["use_pymatgen"] or d["use_forces"]:
            coll_rows = []
            if d["use_pymatgen"] and sim.pymatgen_comparisons > 0:
                rate = 100 * sim.pymatgen_mismatches / sim.pymatgen_comparisons
                coll_rows.append(
                    {
                        "check": "Pymatgen StructureMatcher",
                        "comps": str(sim.pymatgen_comparisons),
                        "agree": str(
                            sim.pymatgen_comparisons - sim.pymatgen_mismatches
                        ),
                        "disagree": str(sim.pymatgen_mismatches),
                        "rate": f"{rate:.1f}%",
                    }
                )
            if d["use_forces"] and sim.force_comparisons > 0:
                rate = 100 * sim.force_mismatches / sim.force_comparisons
                coll_rows.append(
                    {
                        "check": "Force Cosine Similarity",
                        "comps": str(sim.force_comparisons),
                        "agree": str(sim.force_comparisons - sim.force_mismatches),
                        "disagree": str(sim.force_mismatches),
                        "rate": f"{rate:.1f}%",
                    }
                )
            if coll_rows:
                coll_table = SortableTable(
                    columns=_COLLISION_COLS,
                    row_data=coll_rows,
                    format_row=lambda r: [
                        r["check"],
                        r["comps"],
                        r["agree"],
                        r["disagree"],
                        r["rate"],
                    ],
                )
                widgets.extend(
                    [
                        (urwid.Divider(), ("pack", None)),
                        (urwid.Text(("section", " Collision Summary")), ("pack", None)),
                        (urwid.Divider("─"), ("pack", None)),
                        (coll_table, ("given", 6)),
                    ]
                )

        self._results_pile.contents[:] = widgets

    def _on_error(self, error: str) -> None:
        self._running = False
        self._progress_panel.set_finished(False, f"Error: {error}")

    # ------------------------------------------------------------------ #
    #  Save plot
    # ------------------------------------------------------------------ #

    def _save_plot(self) -> None:
        if self._result_data is None or "distances" not in self._result_data:
            return
        from rapmat.core.dedup_analysis import plot_distance_histogram

        d = self._result_data
        plot_path = Path(f"dedup_{d['run_name']}_{d['stage']}.png")
        try:
            plot_distance_histogram(
                d["distances"],
                threshold=d["threshold"],
                save_path=plot_path,
                title=f"Pairwise Distance Distribution — {d['run_name']} ({d['stage']})",
            )
            self._progress_panel.add_log(f"Plot saved to {plot_path}")
        except Exception as e:
            self._progress_panel.add_log(f"Plot error: {e}")

    # ------------------------------------------------------------------ #
    #  Key handling
    # ------------------------------------------------------------------ #

    def keypress(self, size: tuple, key: str) -> str | None:
        if key == "f5":
            self._on_start()
            return None
        if key in ("p", "P"):
            self._save_plot()
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
