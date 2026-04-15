import urwid

from rapmat.tui.widgets.dialog import ModalDialog
from rapmat.tui.widgets.form import (
    FormGroup,
    checkbox_field,
    text_field,
    tuple_field,
)

from rapmat.tui.router import ScreenRouter
from rapmat.tui.state import AppState


class DefectScreen:
    title = "Defect Generation"

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
        self._form = FormGroup(
            fields=[
                text_field(
                    key="structure_file",
                    label="Structure File",
                    default="",
                    validator=lambda v: "Required" if not v.strip() else None,
                ),
                tuple_field(
                    key="supercell",
                    label="Supercell",
                    size=3,
                    default=(4, 4, 1),
                ),
                checkbox_field(
                    key="vacancies",
                    label="Vacancies",
                    default=True,
                ),
                text_field(
                    key="substitutions",
                    label="Substitutions",
                    default="",
                ),
                text_field(
                    key="output_dir",
                    label="Output Dir",
                    default="./defects",
                ),
            ],
            label_width=18,
        )

        self._error_text = urwid.Text("")

        submit_btn = urwid.AttrMap(
            urwid.Button("Generate Defects", on_press=self._on_submit),
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

        help_text = urwid.Text(
            [
                ("form_label", "  Substitutions format: "),
                ("details", "Mo:W,S:Se  (replace Mo→W, S→Se)"),
            ]
        )

        body = urwid.Pile(
            [
                (
                    "pack",
                    urwid.Text(("section", " Generate Point Defects"), align="left"),
                ),
                ("pack", urwid.Divider("─")),
                ("pack", urwid.Divider()),
                ("pack", self._form),
                ("pack", urwid.Divider()),
                ("pack", help_text),
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
        structure_file = vals["structure_file"].strip()
        supercell = vals["supercell"]
        vacancies = vals["vacancies"]
        substitutions_raw = vals["substitutions"].strip()
        output_dir_str = vals["output_dir"].strip() or "./defects"

        try:
            from pathlib import Path

            from ase.io import read as read_ase_structure
            from ase.io import write as write_ase_structure
            from rapmat.core.defect import DefectGenerator

            structure = read_ase_structure(structure_file)
            if isinstance(structure, list):
                structure = structure[-1]

            output_dir = Path(output_dir_str)
            output_dir.mkdir(parents=True, exist_ok=True)

            generator = DefectGenerator(structure)
            all_defects = []

            if vacancies:
                vac_list = generator.generate_vacancies(supercell=supercell)
                all_defects.extend(vac_list)

            if substitutions_raw:
                subs_dict: dict[str, str] = {}
                for part in substitutions_raw.split(","):
                    k, _, v = part.partition(":")
                    if k.strip() and v.strip():
                        subs_dict[k.strip()] = v.strip()
                if subs_dict:
                    sub_list = generator.generate_substitutions(
                        subs_dict, supercell=supercell
                    )
                    all_defects.extend(sub_list)

            if not all_defects:
                self._error_text.set_text(("unconv", "  No defects generated."))
                return

            traj_path = output_dir / "defects.traj"
            write_ase_structure(str(traj_path), [d["atoms"] for d in all_defects])

            formula = structure.get_chemical_formula()
            msg = (
                f"Generated {len(all_defects)} defect(s) for {formula}.\n"
                f"Saved to: {traj_path}"
            )
        except Exception as exc:
            self._error_text.set_text(("form_error", f"  Error: {exc}"))
            return

        if self._frame is not None and self._main_body is not None:
            dlg = ModalDialog.info(
                "Defects Generated",
                msg,
                parent=self._main_body,
                on_close=self._on_info_close,
            )
            self._frame.body = dlg

    def _on_info_close(self) -> None:
        if self._frame is not None and self._main_body is not None:
            self._frame.body = self._main_body

    def keypress(self, size: tuple, key: str) -> str | None:
        if key == "esc":
            self._router.pop()
            return None
        return key
