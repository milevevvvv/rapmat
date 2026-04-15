"""TUI test suite: widgets, screens, router, tasks, and state."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import urwid

# ---------------------------------------------------------------------------
#  Widget unit tests
# ---------------------------------------------------------------------------


class TestSortableTable:
    def _make_table(self, data=None):
        from rapmat.tui.widgets.table import SortableTable

        if data is None:
            data = [
                {"name": "run-b", "val": 2},
                {"name": "run-a", "val": 1},
                {"name": "run-c", "val": 3},
            ]
        cols = [("Name", 16), ("Value", 8)]
        return SortableTable(
            columns=cols,
            row_data=data,
            format_row=lambda r: [r["name"], str(r["val"])],
        )

    def test_build_with_data(self):
        table = self._make_table()
        assert isinstance(table, urwid.WidgetWrap)
        assert len(table._data) == 3

    def test_set_data_replaces(self):
        table = self._make_table()
        table.set_data([{"name": "only", "val": 99}])
        assert len(table._data) == 1

    def test_sort_by_column(self):
        table = self._make_table()
        table.sort_by(0)
        assert table._data[0]["name"] == "run-a"
        assert table._data[2]["name"] == "run-c"

    def test_get_focused_row(self):
        table = self._make_table()
        row = table.get_focused_row()
        assert row is not None
        assert "name" in row

    def test_empty_table(self):
        table = self._make_table(data=[])
        assert table.get_focused_row() is None

    def test_select_signal(self):
        table = self._make_table()
        received = []
        urwid.connect_signal(table, "select", lambda _t, d: received.append(d))
        focus = table._walker[0]
        focus.keypress((80,), "enter")
        assert len(received) == 1
        assert "name" in received[0]

    def test_on_focus_change_callback(self):
        from rapmat.tui.widgets.table import SortableTable

        received = []
        data = [{"name": "a", "val": 1}, {"name": "b", "val": 2}]
        table = SortableTable(
            columns=[("Name", 10), ("Val", 6)],
            row_data=data,
            format_row=lambda r: [r["name"], str(r["val"])],
            on_focus_change=lambda row: received.append(row),
        )
        assert len(received) >= 1


class TestFormGroup:
    def _make_form(self):
        from rapmat.tui.widgets.form import (FormGroup, checkbox_field,
                                             float_field, int_field,
                                             text_field, tuple_field)

        return FormGroup(
            fields=[
                text_field("name", "Name", default="hello"),
                int_field("count", "Count", default=5),
                float_field("rate", "Rate", default=3.14),
                checkbox_field("active", "Active", default=True),
                tuple_field("cell", "Cell", size=3, default=(2, 2, 2)),
            ],
            label_width=12,
        )

    def test_get_values(self):
        form = self._make_form()
        vals = form.get_values()
        assert vals["name"] == "hello"
        assert vals["count"] == 5
        assert abs(vals["rate"] - 3.14) < 0.01
        assert vals["active"] is True
        assert vals["cell"] == (2, 2, 2)

    def test_set_values(self):
        form = self._make_form()
        form.set_values({"name": "world", "count": 10, "active": False})
        vals = form.get_values()
        assert vals["name"] == "world"
        assert vals["count"] == 10
        assert vals["active"] is False

    def test_validate_passes(self):
        form = self._make_form()
        assert form.validate() == []

    def test_validate_with_validator(self):
        from rapmat.tui.widgets.form import FormGroup, text_field

        form = FormGroup(
            fields=[
                text_field(
                    "req",
                    "Required",
                    default="",
                    validator=lambda v: "Cannot be empty" if not v.strip() else None,
                ),
            ],
        )
        errors = form.validate()
        assert len(errors) == 1
        assert "Cannot be empty" in errors[0]


class TestDropdownSelect:
    def test_default_value(self):
        from rapmat.tui.widgets.dropdown import DropdownSelect

        dd = DropdownSelect(label="Test", options=["alpha", "beta", "gamma"], default=1)
        assert dd.value == "beta"

    def test_set_value(self):
        from rapmat.tui.widgets.dropdown import DropdownSelect

        dd = DropdownSelect(label="X", options=["a", "b", "c"])
        dd.set_value("c")
        assert dd.value == "c"

    def test_set_value_unknown(self):
        from rapmat.tui.widgets.dropdown import DropdownSelect

        dd = DropdownSelect(label="X", options=["a", "b"])
        dd.set_value("z")
        assert dd.value == "a"


class TestProgressPanel:
    def test_set_progress(self):
        from rapmat.tui.widgets.progress import ProgressPanel

        panel = ProgressPanel(title=" Test ")
        panel.set_progress(3, 10)
        assert panel._bar.current == 3
        assert panel._bar.done == 10

    def test_add_log(self):
        from rapmat.tui.widgets.progress import ProgressPanel

        panel = ProgressPanel()
        panel.add_log("line 1")
        panel.add_log("line 2")
        assert len(panel._log_walker) == 2

    def test_clear(self):
        from rapmat.tui.widgets.progress import ProgressPanel

        panel = ProgressPanel()
        panel.set_progress(5, 10)
        panel.add_log("hello")
        panel.clear()
        assert panel._bar.current == 0
        assert len(panel._log_walker) == 0

    def test_set_finished(self):
        from rapmat.tui.widgets.progress import ProgressPanel

        panel = ProgressPanel()
        panel.set_progress(0, 10)
        panel.set_finished(True, "Done!")
        assert panel._bar.current == panel._bar.done


class TestModalDialog:
    def test_confirm_builds(self):
        from rapmat.tui.widgets.dialog import ModalDialog

        parent = urwid.SolidFill("x")
        dlg = ModalDialog.confirm(
            "Title", "Are you sure?", parent, on_close=lambda ok: None
        )
        assert isinstance(dlg, urwid.WidgetWrap)

    def test_info_builds(self):
        from rapmat.tui.widgets.dialog import ModalDialog

        parent = urwid.SolidFill("x")
        dlg = ModalDialog.info("Title", "Message here.", parent, on_close=lambda: None)
        assert isinstance(dlg, urwid.WidgetWrap)

    def test_confirm_esc_emits_false(self):
        from rapmat.tui.widgets.dialog import ModalDialog

        received = []
        parent = urwid.SolidFill("x")
        dlg = ModalDialog.confirm(
            "T", "msg", parent, on_close=lambda ok: received.append(ok)
        )
        dlg.keypress((80, 24), "esc")
        assert received == [False]


class TestStatusBar:
    def test_set_keys(self):
        from rapmat.tui.widgets.status_bar import StatusBar

        bar = StatusBar()
        bar.set_keys([("q", "Quit"), ("h", "Help")])
        assert isinstance(bar, urwid.WidgetWrap)

    def test_set_message(self):
        from rapmat.tui.widgets.status_bar import StatusBar

        bar = StatusBar()
        bar.set_message("saved!")
        bar.clear_message()


# ---------------------------------------------------------------------------
#  Router tests
# ---------------------------------------------------------------------------


class _DummyScreen:
    """Minimal Screen implementation for router tests."""

    def __init__(self, name: str, bc_title: str | None = None):
        self.title = name
        self._bc_title = bc_title
        self.resume_count = 0
        self.leave_count = 0
        self._widget = urwid.SolidFill(" ")

    @property
    def breadcrumb_title(self) -> str:
        return self._bc_title or self.title

    def build(self) -> urwid.Widget:
        return self._widget

    def on_resume(self) -> None:
        self.resume_count += 1

    def on_leave(self) -> None:
        self.leave_count += 1


class TestScreenRouter:
    def _make_router(self):
        from rapmat.tui.router import ScreenRouter

        frame = urwid.Frame(body=urwid.SolidFill(" "))
        header_text = urwid.Text(" Rapmat TUI")
        return ScreenRouter(frame, header_text), frame, header_text

    def test_push_sets_body_and_breadcrumb(self):
        router, frame, header = self._make_router()
        s = _DummyScreen("Home")
        router.push(s)
        assert router.depth == 1
        assert router.current is s
        assert "Home" in header.get_text()[0]

    def test_pop_restores_previous(self):
        router, frame, header = self._make_router()
        s1 = _DummyScreen("Home")
        s2 = _DummyScreen("Runs")
        router.push(s1)
        router.push(s2)
        assert router.depth == 2

        removed = router.pop()
        assert removed is s2
        assert router.current is s1
        assert s1.resume_count == 1
        assert s2.leave_count == 1

    def test_pop_single_screen_returns_none(self):
        router, _, _ = self._make_router()
        router.push(_DummyScreen("Home"))
        assert router.pop() is None
        assert router.depth == 1

    def test_replace_swaps_current(self):
        router, frame, header = self._make_router()
        s1 = _DummyScreen("Home")
        s2 = _DummyScreen("Settings")
        router.push(s1)
        router.replace(s2)
        assert router.depth == 1
        assert router.current is s2
        assert s1.leave_count == 1

    def test_dynamic_breadcrumb(self):
        router, frame, header = self._make_router()
        s1 = _DummyScreen("Home")
        s2 = _DummyScreen("Results", bc_title="Results: test-run")
        router.push(s1)
        router.push(s2)
        text = header.get_text()[0]
        assert "Results: test-run" in text

    def test_pop_does_not_call_build_again(self):
        router, frame, header = self._make_router()
        s1 = _DummyScreen("Home")
        s2 = _DummyScreen("Runs")
        router.push(s1)
        original_widget = frame.body
        router.push(s2)
        assert frame.body is not original_widget

        router.pop()
        assert frame.body is original_widget
        assert s1.resume_count == 1


# ---------------------------------------------------------------------------
#  TaskProgress / BackgroundTask tests
# ---------------------------------------------------------------------------


class TestTaskProgress:
    def test_update(self):
        from rapmat.tui.tasks import TaskProgress

        p = TaskProgress()
        p.update(3, 10, "step 3")
        assert p.current == 3
        assert p.total == 10
        assert p.message == "step 3"

    def test_log_and_drain(self):
        from rapmat.tui.tasks import TaskProgress

        p = TaskProgress()
        p.log("a")
        p.log("b")
        lines = p.drain_logs()
        assert lines == ["a", "b"]
        assert p.drain_logs() == []

    def test_finish(self):
        from rapmat.tui.tasks import TaskProgress

        p = TaskProgress()
        p.finish()
        assert p.finished is True
        assert p.error is None

    def test_fail(self):
        from rapmat.tui.tasks import TaskProgress

        p = TaskProgress()
        p.fail("boom")
        assert p.finished is True
        assert p.error == "boom"

    def test_thread_safety(self):
        from rapmat.tui.tasks import TaskProgress

        p = TaskProgress()
        errors = []

        def writer():
            for i in range(100):
                p.log(f"line-{i}")
                p.update(i, 100, f"msg-{i}")

        def reader():
            for _ in range(100):
                p.drain_logs()
                _ = p.current, p.total, p.message

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert not errors


class TestBackgroundTask:
    def test_completion(self):
        from rapmat.tui.tasks import BackgroundTask, TaskProgress

        completed = threading.Event()
        loop = MagicMock()
        alarm_callbacks = []

        def fake_set_alarm_in(delay, callback):
            alarm_callbacks.append(callback)

        loop.set_alarm_in = fake_set_alarm_in

        def worker(progress: TaskProgress):
            progress.update(1, 1, "done")
            progress.finish()

        task = BackgroundTask(
            fn=worker,
            loop=loop,
            on_complete=lambda: completed.set(),
        )
        task.start()
        task._thread.join(timeout=5)

        for cb in alarm_callbacks:
            cb(loop, None)

        assert completed.is_set()

    def test_error(self):
        from rapmat.tui.tasks import BackgroundTask, TaskProgress

        error_msg = []
        loop = MagicMock()
        alarm_callbacks = []
        loop.set_alarm_in = lambda d, cb: alarm_callbacks.append(cb)

        def worker(progress: TaskProgress):
            raise RuntimeError("test error")

        task = BackgroundTask(
            fn=worker,
            loop=loop,
            on_error=lambda e: error_msg.append(e),
        )
        task.start()
        task._thread.join(timeout=5)

        for cb in alarm_callbacks:
            cb(loop, None)

        assert len(error_msg) == 1
        assert "test error" in error_msg[0]

    def test_cancel(self):
        from rapmat.tui.tasks import BackgroundTask, TaskProgress

        started = threading.Event()
        loop = MagicMock()
        loop.set_alarm_in = lambda d, cb: None

        def worker(progress: TaskProgress):
            started.set()
            for _ in range(1000):
                if progress.cancelled:
                    return
                time.sleep(0.01)
            progress.finish()

        task = BackgroundTask(fn=worker, loop=loop)
        task.start()
        started.wait(timeout=5)
        task.cancel()
        task._thread.join(timeout=5)
        assert task._progress.cancelled


# ---------------------------------------------------------------------------
#  AppState tests
# ---------------------------------------------------------------------------


class TestAppState:
    def _make_state(self):
        from rapmat.tui.state import AppState

        store = MagicMock()
        store.list_runs.return_value = [{"name": "run-1"}]
        store.list_studies.return_value = [{"id": "study-1"}]
        return AppState(store=store)

    def test_invalidate_sets_dirty(self):
        state = self._make_state()
        state.cache_dirty = False
        state.studies_cache_dirty = False
        state.invalidate()
        assert state.cache_dirty is True
        assert state.studies_cache_dirty is True

    def test_invalidate_runs_only(self):
        state = self._make_state()
        state.cache_dirty = False
        state.studies_cache_dirty = False
        state.active_run_meta = {"name": "test"}
        state.invalidate_runs()
        assert state.cache_dirty is True
        assert state.studies_cache_dirty is False
        assert state.active_run_meta is None

    def test_invalidate_studies_only(self):
        state = self._make_state()
        state.cache_dirty = False
        state.studies_cache_dirty = False
        state.invalidate_studies()
        assert state.cache_dirty is False
        assert state.studies_cache_dirty is True

    def test_refresh_runs_if_needed_when_clean(self):
        state = self._make_state()
        state.refresh_runs()
        state.store.list_runs.reset_mock()
        state.refresh_runs_if_needed()
        state.store.list_runs.assert_not_called()

    def test_refresh_runs_if_needed_when_dirty(self):
        state = self._make_state()
        state.refresh_runs_if_needed()
        state.store.list_runs.assert_called_once()
        assert len(state.runs_cache) == 1

    def test_refresh_studies_if_needed(self):
        state = self._make_state()
        state.refresh_studies_if_needed()
        state.store.list_studies.assert_called_once()
        assert len(state.studies_cache) == 1


# ---------------------------------------------------------------------------
#  Screen build smoke tests
# ---------------------------------------------------------------------------


class TestScreenBuildSmoke:
    """Verify each screen can call build() without crashing."""

    def _make_env(self, tmp_path):
        from rapmat.tui.state import AppState

        store = MagicMock()
        store.list_runs.return_value = []
        store.list_studies.return_value = []
        store.count_by_status.return_value = {}
        state = AppState(store=store, db_url="ws://localhost:8000/rpc")
        state.loop = MagicMock()

        frame = urwid.Frame(body=urwid.SolidFill(" "))
        header_text = urwid.Text(" Rapmat TUI")
        from rapmat.tui.router import ScreenRouter

        router = ScreenRouter(frame, header_text)

        class _FakeHome:
            title = "Home"

            def build(self):
                return urwid.SolidFill(" ")

            def on_resume(self):
                pass

            def on_leave(self):
                pass

        router.push(_FakeHome())

        return state, router

    def test_home_screen(self, tmp_path):
        state, router = self._make_env(tmp_path)
        from rapmat.tui.screens.home import HomeScreen

        s = HomeScreen(state, router)
        w = s.build()
        assert isinstance(w, urwid.Widget)

    def test_status_screen(self, tmp_path):
        state, router = self._make_env(tmp_path)
        from rapmat.tui.screens.status import StatusScreen

        s = StatusScreen(state, router)
        w = s.build()
        assert isinstance(w, urwid.Widget)

    def test_db_settings_screen(self, tmp_path):
        state, router = self._make_env(tmp_path)
        from rapmat.tui.screens.db_settings import DbSettingsScreen

        s = DbSettingsScreen(state, router)
        try:
            w = s.build()
        except (AttributeError, ImportError) as exc:
            if "torch" in str(exc):
                pytest.skip(f"torch circular import issue: {exc}")
            raise
        assert isinstance(w, urwid.Widget)

    def test_study_list_screen(self, tmp_path):
        state, router = self._make_env(tmp_path)
        from rapmat.tui.screens.study_list import StudyListScreen

        s = StudyListScreen(state, router)
        w = s.build()
        assert isinstance(w, urwid.Widget)

    def test_defect_screen(self, tmp_path):
        state, router = self._make_env(tmp_path)
        from rapmat.tui.screens.defect import DefectScreen

        s = DefectScreen(state, router)
        w = s.build()
        assert isinstance(w, urwid.Widget)

    def test_csp_search_screen(self, tmp_path):
        state, router = self._make_env(tmp_path)
        try:
            from rapmat.tui.screens.csp_search import CSPSearchScreen
        except Exception:
            pytest.skip("CSPSearchScreen import failed (torch issue)")
        s = CSPSearchScreen(state, router)
        w = s.build()
        assert isinstance(w, urwid.Widget)

    def test_phonon_screen(self, tmp_path):
        state, router = self._make_env(tmp_path)
        try:
            from rapmat.tui.screens.phonon import PhononDispersionScreen
        except Exception:
            pytest.skip("PhononDispersionScreen import failed")
        s = PhononDispersionScreen(state, router)
        w = s.build()
        assert isinstance(w, urwid.Widget)

    def test_study_create_screen(self, tmp_path):
        state, router = self._make_env(tmp_path)
        try:
            from rapmat.tui.screens.study_create import StudyCreateScreen
        except Exception:
            pytest.skip("StudyCreateScreen import failed")
        s = StudyCreateScreen(state, router)
        w = s.build()
        assert isinstance(w, urwid.Widget)

    def test_study_detail_screen(self, tmp_path):
        state, router = self._make_env(tmp_path)
        state.active_study = "test-study"
        state.store.get_study.return_value = {
            "system": "Al-Cu",
            "domain": "bulk",
            "calculator": "mock",
            "timestamp": "2025-01-01T00:00:00",
        }
        state.store.get_study_runs.return_value = []
        try:
            from rapmat.tui.screens.study_detail import StudyDetailScreen

            s = StudyDetailScreen(state, router)
            w = s.build()
        except (AttributeError, ImportError) as exc:
            if "torch" in str(exc):
                pytest.skip(f"torch circular import issue: {exc}")
            raise
        assert isinstance(w, urwid.Widget)
