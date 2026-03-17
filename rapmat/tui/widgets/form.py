"""Form widgets for the Rapmat TUI."""

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
    kind: str  # "text" | "int" | "float" | "checkbox" | "radio" | "dropdown" | "tuple"
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
    """Single-line text input."""
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
    """Integer input."""
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
    """Float input (stored as Edit, validated on get_values)."""
    edit = urwid.Edit(caption="", edit_text=str(default))
    return _FieldSpec(
        key=key, label=label, kind="float", widget=edit, validator=validator
    )


def checkbox_field(
    key: str,
    label: str,
    default: bool = False,
) -> _FieldSpec:
    """Boolean checkbox."""
    cb = urwid.CheckBox("", state=bool(default))
    return _FieldSpec(key=key, label=label, kind="checkbox", widget=cb)


def radio_field(
    key: str,
    label: str,
    options: list[str],
    default: int = 0,
) -> _FieldSpec:
    """Radio button group (mutually exclusive)."""
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
    """Popup-based dropdown selector."""
    dd = DropdownSelect(label="", options=options, default=default)
    return _FieldSpec(key=key, label=label, kind="dropdown", widget=dd)


def tuple_field(
    key: str,
    label: str,
    size: int = 3,
    default: tuple = (0, 0, 0),
) -> _FieldSpec:
    """N integer inputs in a row (for supercell/mesh tuples)."""
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
#  FormGroup
# ------------------------------------------------------------------ #


class FormGroup(urwid.WidgetWrap):
    """A ``Pile`` of labeled form fields.

    Parameters
    ----------
    fields:
        List of ``_FieldSpec`` instances created by the factory functions
        above (``text_field``, ``int_field``, etc.).
    label_width:
        Column width reserved for field labels.
    """

    def __init__(self, fields: list[_FieldSpec], label_width: int = 20) -> None:
        self._fields = fields
        self._label_width = label_width
        pile = urwid.Pile(self._build_rows())
        super().__init__(pile)

    # ------------------------------------------------------------------ #
    #  Layout
    # ------------------------------------------------------------------ #

    def _build_rows(self) -> list[urwid.Widget]:
        rows: list[urwid.Widget] = []
        for spec in self._fields:
            label = urwid.Text(("form_label", spec.label + ":"), align="right")
            field_widget = urwid.AttrMap(spec.widget, None, focus_map="focus")
            row = urwid.Columns(
                [
                    (self._label_width, label),
                    ("weight", 1, field_widget),
                ],
                dividechars=1,
            )
            rows.append(row)
        return rows

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def get_values(self) -> dict[str, Any]:
        """Return a dict mapping each field's key to its current value."""
        result: dict[str, Any] = {}
        for spec in self._fields:
            result[spec.key] = self._read_value(spec)
        return result

    def set_values(self, vals: dict[str, Any]) -> None:
        """Batch-set field values by key."""
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
        """Get the underlying urwid widget for a specific field key."""
        for spec in self._fields:
            if spec.key == key:
                return spec.widget
        return None

    def set_field_disabled(self, key: str, disabled: bool) -> None:
        """Dynamically disable or enable a field using urwid.WidgetDisable."""
        for i, spec in enumerate(self._fields):
            if spec.key == key:
                row = self._w.contents[i][0]
                attr_map = row.contents[1][0]
                if disabled and not isinstance(attr_map.original_widget, urwid.WidgetDisable):
                    attr_map.original_widget = urwid.WidgetDisable(spec.widget)
                elif not disabled and isinstance(attr_map.original_widget, urwid.WidgetDisable):
                    attr_map.original_widget = spec.widget
                break

    def validate(self) -> list[str]:
        """Return a list of validation error messages (empty = all valid)."""
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
