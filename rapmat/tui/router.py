import urwid

from typing import Protocol, runtime_checkable


@runtime_checkable
class Screen(Protocol):
    title: str

    def build(self) -> urwid.Widget: ...

    def on_resume(self) -> None: ...

    def on_leave(self) -> None: ...

    @property
    def breadcrumb_title(self) -> str: ...


def _breadcrumb_label(screen: Screen) -> str:
    try:
        return screen.breadcrumb_title
    except AttributeError:
        return screen.title


class ScreenRouter:
    def __init__(self, frame: urwid.Frame, header_text: urwid.Text) -> None:
        self._frame = frame
        self._header_text = header_text
        self._stack: list[tuple[Screen, urwid.Widget]] = []

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    @property
    def current(self) -> Screen | None:
        return self._stack[-1][0] if self._stack else None

    @property
    def depth(self) -> int:
        return len(self._stack)

    def push(self, screen: Screen) -> None:
        if self._stack:
            self._stack[-1][0].on_leave()
        widget = screen.build()
        self._stack.append((screen, widget))
        self._frame.body = widget
        self._update_breadcrumb()

    def pop(self) -> Screen | None:
        if len(self._stack) <= 1:
            return None
        removed_screen, _ = self._stack.pop()
        removed_screen.on_leave()
        current_screen, current_widget = self._stack[-1]
        current_screen.on_resume()
        self._frame.body = current_widget
        self._update_breadcrumb()
        return removed_screen

    def replace(self, screen: Screen) -> None:
        if self._stack:
            self._stack[-1][0].on_leave()
            widget = screen.build()
            self._stack[-1] = (screen, widget)
        else:
            widget = screen.build()
            self._stack.append((screen, widget))
        self._frame.body = widget
        self._update_breadcrumb()

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _update_breadcrumb(self) -> None:
        if not self._stack:
            self._header_text.set_text(" Rapmat TUI")
            return
        path = " > ".join(_breadcrumb_label(s) for s, _w in self._stack)
        self._header_text.set_text(f" {path}")
