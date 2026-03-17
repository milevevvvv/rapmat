"""Screen router for the Rapmat TUI."""

from typing import Protocol, runtime_checkable

import urwid


@runtime_checkable
class Screen(Protocol):
    """Protocol that every TUI screen must satisfy."""

    title: str

    def build(self) -> urwid.Widget:
        """Return the widget tree for this screen.

        Called once when the screen is first pushed and on ``replace()``.
        """
        ...

    def on_resume(self) -> None:
        """Called when this screen becomes active again after a pop.

        Use this to refresh data in-place on the already-built widget.
        ``build()`` will **not** be called again after ``on_resume()``.
        """
        ...

    def on_leave(self) -> None:
        """Called just before this screen is replaced or pushed away from."""
        ...

    @property
    def breadcrumb_title(self) -> str:
        """Label shown in the breadcrumb header.

        Defaults to ``title`` but screens may override to include
        dynamic context (e.g. the active run name).
        """
        ...


def _breadcrumb_label(screen: Screen) -> str:
    """Return the breadcrumb label, falling back to ``title``."""
    try:
        return screen.breadcrumb_title
    except AttributeError:
        return screen.title


class ScreenRouter:
    """Push/pop navigation stack.

    Parameters
    ----------
    frame:
        The top-level ``urwid.Frame`` whose ``body`` is swapped on
        every navigation event.
    header_text:
        A ``urwid.Text`` widget embedded in the header.  The router
        writes the breadcrumb path into it on every navigation event.
    """

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
        """Push *screen* onto the stack and display it."""
        if self._stack:
            self._stack[-1][0].on_leave()
        widget = screen.build()
        self._stack.append((screen, widget))
        self._frame.body = widget
        self._update_breadcrumb()

    def pop(self) -> Screen | None:
        """Remove the top screen and return to the previous one.

        The previous screen's ``on_resume()`` is called so it can
        refresh data in-place; the cached widget is restored directly
        without calling ``build()`` again.

        Returns the screen that was removed, or ``None`` if the stack
        had only one entry (home screen -- cannot pop further).
        """
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
        """Replace the current screen without growing the stack."""
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
        """Rewrite the header breadcrumb from the current stack titles."""
        if not self._stack:
            self._header_text.set_text(" Rapmat TUI")
            return
        path = " > ".join(_breadcrumb_label(s) for s, _w in self._stack)
        self._header_text.set_text(f" {path}")
