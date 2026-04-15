import urwid


class StatusBar(urwid.WidgetWrap):
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
        parts = [f"[{k}] {desc}" for k, desc in keys]
        self._keys_text.set_text("  " + "  ".join(parts))

    def set_message(self, msg: str) -> None:
        self._msg_text.set_text(msg + " ")

    def clear_message(self) -> None:
        self._msg_text.set_text("")
