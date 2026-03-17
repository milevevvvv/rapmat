"""Dropdown selector widget for the Rapmat TUI."""

import urwid


class _PopupBody(urwid.WidgetWrap):
    """Thin wrapper that closes the parent popup on Escape."""

    def __init__(self, widget: urwid.Widget, close_fn) -> None:
        self._close = close_fn
        super().__init__(widget)

    def keypress(self, size: tuple, key: str) -> str | None:
        if key == "esc":
            self._close()
            return None
        return super().keypress(size, key)


class DropdownSelect(urwid.PopUpLauncher):
    """Pick-one dropdown built on urwid's popup overlay system.

    Displays the current selection as a button; clicking opens a popup
    list with radio-style markers (● / ○).  Emits a ``"change"`` signal
    with the new value string.

    Parameters
    ----------
    label:
        Label prefix shown on the button (e.g. ``"Domain"``).
    options:
        Sequence of string choices.
    default:
        Index of the initially selected option.
    """

    signals = ["change"]

    def __init__(self, label: str, options: list[str], default: int = 0) -> None:
        self.options = list(options)
        self.selected = default
        self.label = label
        self._btn = urwid.Button(self._text())
        urwid.connect_signal(self._btn, "click", lambda b: self.open_pop_up())
        super().__init__(self._btn)

    def _text(self) -> str:
        return f"{self.label} \u25be  {self.options[self.selected]}"

    def create_pop_up(self) -> urwid.Widget:
        items = []
        for i, opt in enumerate(self.options):
            marker = "\u25cf " if i == self.selected else "\u25cb "
            btn = urwid.Button(marker + opt)
            urwid.connect_signal(btn, "click", self._pick, i)
            items.append(urwid.AttrMap(btn, None, focus_map="dropdown_hl"))
        lb = urwid.ListBox(urwid.SimpleListWalker(items))
        return _PopupBody(urwid.LineBox(lb), self.close_pop_up)

    def _pick(self, btn: urwid.Button, idx: int) -> None:
        self.selected = idx
        self._btn.set_label(self._text())
        self.close_pop_up()
        self._emit("change", self.options[idx])

    def get_pop_up_parameters(self) -> dict:
        w = max(len(o) for o in self.options) + 14
        h = min(len(self.options) + 2, 10)
        return {"left": 0, "top": 1, "overlay_width": w, "overlay_height": h}

    @property
    def value(self) -> str:
        """Currently selected option string."""
        return self.options[self.selected]

    def set_value(self, value: str) -> None:
        """Set the selection by value string (no-op if not found)."""
        try:
            idx = self.options.index(value)
        except ValueError:
            return
        self.selected = idx
        self._btn.set_label(self._text())
