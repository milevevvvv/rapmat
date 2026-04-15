import urwid

from typing import Callable, Sequence


class ModalDialog(urwid.WidgetWrap):
    signals = ["close"]

    def __init__(
        self,
        title: str,
        body_widget: urwid.Widget,
        parent: urwid.Widget,
    ) -> None:
        self._parent = parent

        inner = urwid.LineBox(
            urwid.Padding(body_widget, left=1, right=1),
            title=title,
        )
        overlay = urwid.Overlay(
            urwid.AttrMap(inner, "dialog"),
            parent,
            align=urwid.CENTER,
            width=(urwid.RELATIVE, 50),
            valign=urwid.MIDDLE,
            height=urwid.PACK,
            min_width=40,
        )
        super().__init__(overlay)

    # ------------------------------------------------------------------ #
    #  Static constructors
    # ------------------------------------------------------------------ #

    @staticmethod
    def confirm(
        title: str,
        message: str,
        parent: urwid.Widget,
        on_close: Callable[[bool], None],
    ) -> "ModalDialog":
        dlg: ModalDialog | None = None

        def _yes(btn: urwid.Button) -> None:
            assert dlg is not None
            dlg._emit("close", True)
            on_close(True)

        def _no(btn: urwid.Button) -> None:
            assert dlg is not None
            dlg._emit("close", False)
            on_close(False)

        yes_btn = urwid.AttrMap(
            urwid.Button("Yes", on_press=_yes), None, focus_map="btn_focus"
        )
        no_btn = urwid.AttrMap(
            urwid.Button("No", on_press=_no), None, focus_map="btn_focus"
        )

        body = urwid.Pile(
            [
                urwid.Text(message),
                urwid.Divider(),
                urwid.Columns(
                    [("weight", 1, yes_btn), ("weight", 1, no_btn)],
                    dividechars=2,
                ),
            ]
        )
        dlg = ModalDialog(title, body, parent)

        original_keypress = dlg.keypress

        def _keypress(size: tuple, key: str) -> str | None:
            if key == "esc":
                dlg._emit("close", False)
                on_close(False)
                return None
            return original_keypress(size, key)

        dlg.keypress = _keypress  # type: ignore[method-assign]
        return dlg

    @staticmethod
    def info(
        title: str,
        message: str,
        parent: urwid.Widget,
        on_close: Callable[[], None],
    ) -> "ModalDialog":
        dlg: ModalDialog | None = None

        def _ok(btn: urwid.Button) -> None:
            assert dlg is not None
            dlg._emit("close", True)
            on_close()

        ok_btn = urwid.AttrMap(
            urwid.Button("OK", on_press=_ok), None, focus_map="btn_focus"
        )

        body = urwid.Pile(
            [
                urwid.Text(message),
                urwid.Divider(),
                urwid.Padding(ok_btn, align=urwid.CENTER, width=10),
            ]
        )
        dlg = ModalDialog(title, body, parent)

        original_keypress = dlg.keypress

        def _keypress_info(size: tuple, key: str) -> str | None:
            if key == "esc":
                dlg._emit("close", True)
                on_close()
                return None
            return original_keypress(size, key)

        dlg.keypress = _keypress_info  # type: ignore[method-assign]
        return dlg

    @staticmethod
    def error(
        title: str,
        message: str,
        parent: urwid.Widget,
        actions: Sequence[tuple[str, Callable[[], None]]],
        esc_action_index: int = -1,
    ) -> "ModalDialog":
        dlg: ModalDialog | None = None

        def _make_handler(cb: Callable[[], None]):
            def _press(_btn: urwid.Button) -> None:
                assert dlg is not None
                dlg._emit("close", True)
                cb()

            return _press

        buttons = []
        for label, cb in actions:
            btn = urwid.AttrMap(
                urwid.Button(label, on_press=_make_handler(cb)),
                None,
                focus_map="btn_focus",
            )
            buttons.append(("weight", 1, btn))

        body = urwid.Pile(
            [
                urwid.Text(message),
                urwid.Divider(),
                urwid.Columns(buttons, dividechars=2),
            ]
        )
        dlg = ModalDialog(title, body, parent)

        esc_idx = esc_action_index if esc_action_index >= 0 else len(actions) - 1
        esc_cb = actions[esc_idx][1] if 0 <= esc_idx < len(actions) else None

        original_keypress = dlg.keypress

        def _keypress(size: tuple, key: str) -> str | None:
            if key == "esc" and esc_cb is not None:
                assert dlg is not None
                dlg._emit("close", True)
                esc_cb()
                return None
            return original_keypress(size, key)

        dlg.keypress = _keypress  # type: ignore[method-assign]
        return dlg

    @staticmethod
    def input_text(
        title: str,
        message: str,
        parent: urwid.Widget,
        on_save: "Callable[[str], None]",
        on_cancel: "Callable[[], None] | None" = None,
        default: str = "",
    ) -> "ModalDialog":
        dlg: ModalDialog | None = None
        edit = urwid.Edit(caption="  ", edit_text=default)

        def _save(btn: urwid.Button) -> None:
            assert dlg is not None
            dlg._emit("close", True)
            on_save(edit.get_edit_text())

        def _cancel(btn: urwid.Button) -> None:
            assert dlg is not None
            dlg._emit("close", False)
            if on_cancel:
                on_cancel()

        save_btn = urwid.AttrMap(
            urwid.Button("Save", on_press=_save), None, focus_map="btn_focus"
        )
        cancel_btn = urwid.AttrMap(
            urwid.Button("Cancel", on_press=_cancel), None, focus_map="btn_focus"
        )

        body = urwid.Pile(
            [
                urwid.Text(message),
                urwid.Divider(),
                urwid.AttrMap(edit, "input"),
                urwid.Divider(),
                urwid.Columns(
                    [("weight", 1, save_btn), ("weight", 1, cancel_btn)],
                    dividechars=2,
                ),
            ]
        )
        dlg = ModalDialog(title, body, parent)

        original_keypress = dlg.keypress

        def _keypress(size: tuple, key: str) -> str | None:
            if key == "esc":
                assert dlg is not None
                dlg._emit("close", False)
                if on_cancel:
                    on_cancel()
                return None
            return original_keypress(size, key)

        dlg.keypress = _keypress
        return dlg
