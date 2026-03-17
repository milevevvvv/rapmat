"""Context-sensitive footer status bar for the Rapmat TUI."""

import urwid


class StatusBar(urwid.WidgetWrap):
    """Footer widget that displays context-sensitive key hints and messages.

    Usage::

        bar = StatusBar()
        bar.set_keys([("q", "Quit"), ("Enter", "Open"), ("/", "Search")])
        bar.set_message("Saved to structure_1.cif")
        bar.clear_message()
    """

    def __init__(self) -> None:
        self._keys_text = urwid.Text("", wrap="clip")
        self._msg_text = urwid.Text("", align="right", wrap="clip")
        cols = urwid.Columns(
            [
                ("weight", 3, self._keys_text),
                ("weight", 1, self._msg_text),
            ],
            dividechars=1,
        )
        super().__init__(urwid.AttrMap(cols, "footer"))

    def set_keys(self, keys: list[tuple[str, str]]) -> None:
        """Set the key-hint section.

        Parameters
        ----------
        keys:
            List of ``(key, description)`` pairs, e.g.
            ``[("q", "Quit"), ("Enter", "Open")]``.
            Rendered as: ``[q] Quit  [Enter] Open``
        """
        parts = [f"[{k}] {desc}" for k, desc in keys]
        self._keys_text.set_text("  " + "  ".join(parts))

    def set_message(self, msg: str) -> None:
        """Display a transient message on the right side of the bar."""
        self._msg_text.set_text(msg + " ")

    def clear_message(self) -> None:
        """Clear the transient message."""
        self._msg_text.set_text("")
