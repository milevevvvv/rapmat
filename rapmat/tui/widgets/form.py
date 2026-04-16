from dataclasses import dataclass, field
from typing import Any

import urwid

from rapmat.tui.widgets.dropdown import DropdownSelect

# ------------------------------------------------------------------ #
#  Internal field descriptors
# ------------------------------------------------------------------ #


@dataclass
class _FieldSpec:
    key: str
    label: str
    kind: str
    widget: urwid.Widget
    radio_buttons: list[urwid.RadioButton] = field(default_factory=list)
    int_edits: list[urwid.IntEdit] = field(default_factory=list)
    validator: Any = None


# ------------------------------------------------------------------ #
#  Factory helpers
# ------------------------------------------------------------------ #


def text_field(
    key: str,
    label: str,
    default: str = "",
    validator=None,
) -> _FieldSpec:
    edit = urwid.Edit(caption="", edit_text=str(default))
    return _FieldSpec(
        key=key, label=label, kind="text", widget=edit, validator=validator
    )


def int_field(
    key: str,
    label: str,
    default: int = 0,
    validator=None,
) -> _FieldSpec:
    edit = urwid.IntEdit(default=int(default))
    return _FieldSpec(
        key=key, label=label, kind="int", widget=edit, validator=validator
    )


def float_field(
    key: str,
    label: str,
    default: float = 0.0,
    validator=None,
) -> _FieldSpec:
    edit = urwid.Edit(caption="", edit_text=str(default))
    return _FieldSpec(
        key=key, label=label, kind="float", widget=edit, validator=validator
    )


def checkbox_field(
    key: str,
    label: str,
    default: bool = False,
) -> _FieldSpec:
    cb = urwid.CheckBox("", state=bool(default))
    return _FieldSpec(key=key, label=label, kind="checkbox", widget=cb)


def radio_field(
    key: str,
    label: str,
    options: list[str],
    default: int = 0,
) -> _FieldSpec:
    group: list[urwid.RadioButton] = []
    for i, opt in enumerate(options):
        rb = urwid.RadioButton(group, opt, state=(i == default))
        _ = rb  # suppress unused warning
    pile = urwid.Pile(group)
    return _FieldSpec(
        key=key, label=label, kind="radio", widget=pile, radio_buttons=group
    )


def dropdown_field(
    key: str,
    label: str,
    options: list[str],
    default: int = 0,
) -> _FieldSpec:
    dd = DropdownSelect(label="", options=options, default=default)
    return _FieldSpec(key=key, label=label, kind="dropdown", widget=dd)


def tuple_field(
    key: str,
    label: str,
    size: int = 3,
    default: tuple = (0, 0, 0),
) -> _FieldSpec:
    edits = [
        urwid.IntEdit(default=int(default[i]) if i < len(default) else 0)
        for i in range(size)
    ]
    cols = urwid.Columns(
        [(6, urwid.AttrMap(e, None, focus_map="focus")) for e in edits],
        dividechars=1,
    )
    return _FieldSpec(key=key, label=label, kind="tuple", widget=cols, int_edits=edits)


# ------------------------------------------------------------------ #
#  Focus-aware group box
# ------------------------------------------------------------------ #


def create_focus_group(title: str, content: urwid.Widget) -> urwid.Widget:
    """Wrap *content* in a LineBox that highlights when any child is focused."""
    inner = urwid.AttrMap(content, "body")
    box = urwid.LineBox(
        inner,
        title=title,
        title_attr="dim_title",
        title_align="center",
    )
    return urwid.AttrMap(
        box,
        "dim_border",
        focus_map={
            None: "focus_border",
            "dim_title": "focus_title",
        },
    )


# ------------------------------------------------------------------ #
#  FormGroup
# ------------------------------------------------------------------ #


class FormGroup(urwid.WidgetWrap):
    """A form with optional focus-aware group sections.

    Parameters
    ----------
    fields:
        Flat list of ``_FieldSpec`` objects.
    label_width:
        Fixed column width for the label text.
    groups:
        Optional list of ``(title, [field_key, ...])`` tuples.  When
        provided, fields are organised into named LineBox groups that
        visually highlight on focus.  Any field key **not** listed in
        *groups* is appended to a trailing unnamed section.
    """

    def __init__(
        self,
        fields: list[_FieldSpec],
        label_width: int = 20,
        groups: list[tuple[str, list[str]]] | None = None,
    ) -> None:
        self._fields = fields
        self._label_width = label_width
        self._row_by_key: dict[str, urwid.Columns] = {}
        pile = urwid.Pile(self._build_rows(groups))
        super().__init__(pile)

    # ------------------------------------------------------------------ #
    #  Layout
    # ------------------------------------------------------------------ #

    def _make_row(self, spec: _FieldSpec) -> urwid.Columns:
        label = urwid.Text(("form_label", spec.label + ":"), align="right")
        field_widget = urwid.AttrMap(spec.widget, None, focus_map="focus")
        row = urwid.Columns(
            [
                (self._label_width, label),
                ("weight", 1, field_widget),
            ],
            dividechars=1,
        )
        self._row_by_key[spec.key] = row
        return row

    def _build_rows(
        self, groups: list[tuple[str, list[str]]] | None = None
    ) -> list[urwid.Widget]:
        spec_by_key = {s.key: s for s in self._fields}

        if groups is None:
            return [self._make_row(s) for s in self._fields]

        widgets: list[urwid.Widget] = []
        used_keys: set[str] = set()

        for title, keys in groups:
            group_rows: list[urwid.Widget] = []
            for key in keys:
                spec = spec_by_key.get(key)
                if spec is None:
                    continue
                group_rows.append(self._make_row(spec))
                used_keys.add(key)
            if group_rows:
                pile = urwid.Pile(group_rows)
                widgets.append(create_focus_group(title, pile))
                widgets.append(urwid.Divider())

        leftovers = [s for s in self._fields if s.key not in used_keys]
        if leftovers:
            leftover_rows = [self._make_row(s) for s in leftovers]
            pile = urwid.Pile(leftover_rows)
            widgets.append(create_focus_group("Other", pile))

        return widgets

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def get_values(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for spec in self._fields:
            result[spec.key] = self._read_value(spec)
        return result

    def set_values(self, vals: dict[str, Any]) -> None:
        for spec in self._fields:
            if spec.key not in vals:
                continue
            v = vals[spec.key]
            if spec.kind == "text":
                spec.widget.set_edit_text(str(v))
            elif spec.kind == "int":
                spec.widget.set_edit_text(str(int(v)))
            elif spec.kind == "float":
                spec.widget.set_edit_text(str(v))
            elif spec.kind == "checkbox":
                spec.widget.set_state(bool(v))
            elif spec.kind == "radio":
                if isinstance(v, int):
                    for i, rb in enumerate(spec.radio_buttons):
                        rb.set_state(i == v, do_callback=False)
                else:
                    for rb in spec.radio_buttons:
                        rb.set_state(rb.label == str(v), do_callback=False)
            elif spec.kind == "dropdown":
                spec.widget.set_value(str(v))
            elif spec.kind == "tuple":
                for i, edit in enumerate(spec.int_edits):
                    val_i = v[i] if hasattr(v, "__getitem__") and i < len(v) else 0
                    edit.set_edit_text(str(int(val_i)))

    def get_widget(self, key: str) -> urwid.Widget | None:
        for spec in self._fields:
            if spec.key == key:
                return spec.widget
        return None

    def set_field_disabled(self, key: str, disabled: bool) -> None:
        for spec in self._fields:
            if spec.key == key:
                row = self._row_by_key.get(key)
                if row is None:
                    break
                attr_map = row.contents[1][0]
                if disabled and not isinstance(
                    attr_map.original_widget, urwid.WidgetDisable
                ):
                    attr_map.original_widget = urwid.WidgetDisable(spec.widget)
                elif not disabled and isinstance(
                    attr_map.original_widget, urwid.WidgetDisable
                ):
                    attr_map.original_widget = spec.widget
                break

    def validate(self) -> list[str]:
        errors: list[str] = []
        for spec in self._fields:
            raw = self._read_value(spec)
            if spec.kind == "float":
                try:
                    float(spec.widget.get_edit_text())
                except ValueError:
                    errors.append(f"{spec.label}: must be a number")
            if spec.validator is not None:
                msg = spec.validator(raw)
                if msg:
                    errors.append(f"{spec.label}: {msg}")
        return errors

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    def _read_value(self, spec: _FieldSpec) -> Any:
        if spec.kind == "text":
            return spec.widget.get_edit_text()
        if spec.kind == "int":
            return spec.widget.value()
        if spec.kind == "float":
            try:
                return float(spec.widget.get_edit_text())
            except ValueError:
                return 0.0
        if spec.kind == "checkbox":
            return spec.widget.get_state()
        if spec.kind == "radio":
            for rb in spec.radio_buttons:
                if rb.state:
                    return rb.label
            return spec.radio_buttons[0].label if spec.radio_buttons else ""
        if spec.kind == "dropdown":
            return spec.widget.value
        if spec.kind == "tuple":
            return tuple(e.value() for e in spec.int_edits)
        return None
