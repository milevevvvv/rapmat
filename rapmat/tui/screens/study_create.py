"""Study create screen for the Rapmat TUI."""

import urwid

from rapmat.tui.widgets.dialog import ModalDialog
from rapmat.tui.widgets.dropdown import DropdownSelect
from rapmat.tui.widgets.form import (
    FormGroup,
    checkbox_field,
    dropdown_field,
    float_field,
    int_field,
    text_field,
)

from rapmat.tui.router import ScreenRouter
from rapmat.tui.state import AppState


def _get_calculator_options() -> list[str]:
    from rapmat.calculators import Calculators

    return [c.value for c in Calculators]


class StudyCreateScreen:
    """Create a new phase diagram study."""

    title = "New Study"

    def __init__(self, state: "AppState", router: "ScreenRouter") -> None:
        self._state = state
        self._router = router
        self._form: FormGroup | None = None
        self._error_text: urwid.Text | None = None
        self._frame: urwid.Frame | None = None
        self._main_body: urwid.Widget | None = None

    # ------------------------------------------------------------------ #
    #  Screen protocol
    # ------------------------------------------------------------------ #

    def build(self) -> urwid.Widget:
        calc_options = _get_calculator_options()

        self._form = FormGroup(
            fields=[
                text_field(
                    key="system",
                    label="System (e.g., Al-O)",
                    default="",
                    validator=lambda v: "Required" if not v.strip() else None,
                ),
                text_field(
                    key="name",
                    label="Study Name",
                    default="",
                    validator=lambda v: "Required" if not v.strip() else None,
                ),
                dropdown_field(
                    key="domain",
                    label="Domain",
                    options=["bulk", "monolayer"],
                    default=0,
                ),
                dropdown_field(
                    key="calculator",
                    label="Calculator",
                    options=calc_options,
                    default=0,
                ),
                text_field(
                    key="calculator_config",
                    label="Config TOML Path",
                    default="",
                ),
                float_field("pressure", "Pressure (GPa)", default=0.0),
                # Convergence section
                float_field("force_conv_crit", "Force conv. crit", default=5e-2),
                int_field("steps_max", "Max steps", default=500),
                # Dedup section
                checkbox_field("dedup", "Dedup", default=False),
                float_field("dedup_threshold", "Dedup threshold", default=1e-2),
                checkbox_field("pymatgen_dedup", "Pymatgen dedup", default=False),
                checkbox_field("force_dedup", "Force dedup", default=False),
                float_field("force_cosine", "Force cosine thresh", default=0.95),
                # Sanity section
                float_field("min_dist", "Min distance (Å)", default=0.5),
                checkbox_field("sanity_pymatgen", "Sanity pymatgen", default=False),
                float_field("symprec", "Symmetry precision", default=1e-5),
            ],
            label_width=22,
        )

        self._error_text = urwid.Text("")

        submit_btn = urwid.AttrMap(
            urwid.Button("Create Study", on_press=self._on_submit),
            None,
            focus_map="btn_focus",
        )
        cancel_btn = urwid.AttrMap(
            urwid.Button("Cancel", on_press=lambda _: self._router.pop()),
            None,
            focus_map="btn_focus",
        )
        btn_row = urwid.Columns(
            [("weight", 1, submit_btn), ("weight", 1, cancel_btn)],
            dividechars=2,
        )

        body = urwid.Pile(
            [
                ("pack", urwid.Divider()),
                ("pack", self._form),
                ("pack", urwid.Divider()),
                ("pack", self._error_text),
                ("pack", urwid.Divider()),
                ("pack", btn_row),
            ]
        )

        listbox = urwid.ListBox(urwid.SimpleListWalker([body]))
        scrollable = urwid.ScrollBar(
            listbox,
            trough_char=urwid.ScrollBar.Symbols.LITE_SHADE,
        )
        self._main_body = urwid.Padding(scrollable, left=2, right=2)
        self._frame = urwid.Frame(body=self._main_body)
        self._update_footer()
        return self._frame

    def on_resume(self) -> None:
        self._update_footer()

    def on_leave(self) -> None:
        pass

    def _update_footer(self) -> None:
        if self._state.status_bar:
            self._state.status_bar.set_keys(
                [
                    ("Tab", "Navigate"),
                    ("Enter", "Submit"),
                    ("Esc", "Cancel"),
                ]
            )
            self._state.status_bar.clear_message()

    def on_leave(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    #  Submit
    # ------------------------------------------------------------------ #

    def _on_submit(self, _btn) -> None:
        if self._form is None or self._error_text is None:
            return

        errors = self._form.validate()
        if errors:
            self._error_text.set_text(("form_error", "  " + "; ".join(errors)))
            return

        vals = self._form.get_values()
        system_raw = vals["system"].strip()
        name = vals["name"].strip()
        domain = vals["domain"]
        calculator = vals["calculator"]
        calculator_config_path = vals.get("calculator_config", "").strip()

        calc_config_dict = {}
        if calculator_config_path:
            import tomllib
            from pathlib import Path
            config_file = Path(calculator_config_path)
            if not config_file.is_file():
                self._error_text.set_text(("form_error", f"  Config file not found: {calculator_config_path}"))
                return
            try:
                with open(config_file, "rb") as f:
                    calc_config_dict = tomllib.load(f)
            except Exception as e:
                self._error_text.set_text(("form_error", f"  Invalid TOML in config: {e}"))
                return

        try:
            from rapmat.utils.common import format_system, parse_system

            elements = parse_system(system_raw)
            normalized_system = format_system(elements)
        except Exception as exc:
            self._error_text.set_text(("form_error", f"  Invalid system: {exc}"))
            return

        try:
            self._state.store.create_study(
                study_id=name,
                system=normalized_system,
                domain=domain,
                calculator=calculator,
                config={
                    "calculator_config": calc_config_dict,
                    "pressure_gpa": vals["pressure"],
                    "force_conv_crit": vals["force_conv_crit"],
                    "steps_max": vals["steps_max"],
                    "dedup": vals["dedup"],
                    "dedup_threshold": vals["dedup_threshold"],
                    "pymatgen_dedup": vals["pymatgen_dedup"],
                    "pymatgen_ltol": 0.2,
                    "pymatgen_stol": 0.3,
                    "pymatgen_angle_tol": 5.0,
                    "force_dedup": vals["force_dedup"],
                    "force_cosine_threshold": vals["force_cosine"],
                    "min_dist": vals["min_dist"],
                    "sanity_pymatgen": vals["sanity_pymatgen"],
                    "sanity_pymatgen_tol": 0.5,
                    "symprec": vals["symprec"],
                }
            )
        except Exception as exc:
            self._error_text.set_text(("form_error", f"  Error: {exc}"))
            return

        self._state.invalidate()

        if self._frame is not None and self._main_body is not None:
            dlg = ModalDialog.info(
                "Study Created",
                f"Study '{name}' created.\nSystem: {normalized_system}\nDomain: {domain}",
                parent=self._main_body,
                on_close=self._on_info_close,
            )
            self._frame.body = dlg

    def _on_info_close(self) -> None:
        self._router.pop()

    def keypress(self, size: tuple, key: str) -> str | None:
        if key == "esc":
            self._router.pop()
            return None
        return key
