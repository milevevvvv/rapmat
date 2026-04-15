import urwid


class ProgressPanel(urwid.WidgetWrap):
    def __init__(self, title: str = " Progress ") -> None:
        self._bar = urwid.ProgressBar("progress", "pg_done", current=0, done=100)
        self._log_walker = urwid.SimpleListWalker([])
        self._log_box = urwid.ListBox(self._log_walker)
        self._status_text = urwid.Text("", align="center")

        body = urwid.Pile(
            [
                ("pack", self._bar),
                ("pack", urwid.Divider()),
                ("pack", self._status_text),
                ("pack", urwid.Divider()),
                ("weight", 1, self._log_box),
            ]
        )
        outer = urwid.LineBox(urwid.Padding(body, left=1, right=1), title=title)
        super().__init__(outer)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def set_progress(self, current: int, total: int, message: str = "") -> None:
        if total <= 0:
            total = 1
        self._bar.done = total
        self._bar.set_completion(current)

        if message:
            self._status_text.set_text(message)
        else:
            self._status_text.set_text(f"{current} / {total}")

    def add_log(self, message: str) -> None:
        self._log_walker.append(urwid.Text(("log_line", message)))
        self._log_box.set_focus(len(self._log_walker) - 1)

    def clear(self) -> None:
        self._bar.set_completion(0)
        self._status_text.set_text("")
        self._log_walker[:] = []

    def set_cancelling(self) -> None:
        self._status_text.set_text(("error", "⏳ Awaiting cancellation…"))
        self.add_log("Cancellation requested, waiting for current operation to finish…")

    def set_finished(self, success: bool, message: str) -> None:
        attr = "success" if success else "error"
        self._status_text.set_text((attr, message))
        if success:
            self._bar.set_completion(self._bar.done)
