from typing import Callable

import urwid


class SelectableRow(urwid.WidgetWrap):
    signals = ["select"]

    def __init__(
        self,
        data: dict,
        texts: list[str],
        widths: list[int],
        attr: str = "body",
    ) -> None:
        self.data = data

        cols = [
            (width, urwid.Text(text, wrap="clip")) for text, width in zip(texts, widths)
        ]
        row_widget = urwid.Columns(cols, dividechars=1)
        super().__init__(urwid.AttrMap(row_widget, attr, focus_map="focus"))

    def selectable(self) -> bool:
        return True

    def keypress(self, size: tuple, key: str) -> str | None:
        if key == "enter":
            self._emit("select", self.data)
            return None
        return key


class _TableListBox(urwid.ListBox):
    def __init__(
        self, walker: urwid.SimpleFocusListWalker, table: "SortableTable"
    ) -> None:
        super().__init__(walker)
        self._table = table

    def keypress(self, size: tuple, key: str) -> str | None:
        result = super().keypress(size, key)
        self._table._on_focus_change()
        return result

    def mouse_event(self, size, event, button, col, row, focus):
        result = super().mouse_event(size, event, button, col, row, focus)
        self._table._on_focus_change()
        return result


class SortableTable(urwid.WidgetWrap):
    signals = ["select"]

    def __init__(
        self,
        columns: list[tuple[str, int]],
        row_data: list[dict],
        format_row: Callable[[dict], list[str]],
        attr_fn: Callable[[dict], str] | None = None,
        on_focus_change: Callable[[dict | None], None] | None = None,
    ) -> None:
        self._columns = columns
        self._format_row = format_row
        self._attr_fn = attr_fn or (lambda _: "body")
        self._on_focus_change_cb = on_focus_change
        self._data: list[dict] = []

        widths = [w for _, w in columns]
        titles = [t for t, _ in columns]

        header_cols = [
            (width, urwid.Text(title, wrap="clip"))
            for title, width in zip(titles, widths)
        ]
        self._col_header = urwid.AttrMap(
            urwid.Columns(header_cols, dividechars=1), "col_header"
        )

        self._walker = urwid.SimpleFocusListWalker([])
        self._listbox = _TableListBox(self._walker, self)

        scrollable_listbox = urwid.ScrollBar(
            self._listbox,
            trough_char=urwid.ScrollBar.Symbols.LITE_SHADE,
        )

        body = urwid.Frame(
            body=urwid.AttrMap(scrollable_listbox, "body"),
            header=self._col_header,
        )
        super().__init__(body)

        self.set_data(row_data)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def set_data(self, rows: list[dict]) -> None:
        self._data = list(rows)
        self._rebuild_walker()

    def update_columns(self, columns: list[tuple[str, int]]) -> None:
        self._columns = columns
        widths = [w for _, w in columns]
        titles = [t for t, _ in columns]

        header_cols = [
            (width, urwid.Text(title, wrap="clip"))
            for title, width in zip(titles, widths)
        ]
        self._col_header = urwid.AttrMap(
            urwid.Columns(header_cols, dividechars=1), "col_header"
        )
        self._w.header = self._col_header

        self._rebuild_walker()

    def get_focused_row(self) -> dict | None:
        widget = self._listbox.focus
        if widget is not None and isinstance(widget, SelectableRow):
            return widget.data
        return None

    def sort_by(self, col_index: int, reverse: bool = False) -> None:
        def _key(row: dict) -> str:
            texts = self._format_row(row)
            if col_index < len(texts):
                return texts[col_index].lower()
            return ""

        self._data.sort(key=_key, reverse=reverse)
        self._rebuild_walker()

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _rebuild_walker(self) -> None:
        widths = [w for _, w in self._columns]
        widgets: list[urwid.Widget] = []
        for row in self._data:
            texts = self._format_row(row)
            attr = self._attr_fn(row)
            row_widget = SelectableRow(row, texts, widths, attr)
            urwid.connect_signal(row_widget, "select", self._on_row_select)
            widgets.append(row_widget)

        self._walker[:] = widgets

        if widgets:
            try:
                self._listbox.set_focus(0)
            except Exception:
                pass
            self._on_focus_change()
        else:
            if self._on_focus_change_cb:
                self._on_focus_change_cb(None)

    def _on_row_select(self, widget: SelectableRow, data: dict) -> None:
        self._emit("select", data)

    def _on_focus_change(self) -> None:
        if self._on_focus_change_cb:
            self._on_focus_change_cb(self.get_focused_row())
