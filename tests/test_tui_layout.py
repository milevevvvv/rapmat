import sys

import urwid


def test_tui():
    from rapmat.db_config import resolve_store
    from rapmat.tui.app import RapmatApp
    from rapmat.tui.state import AppState

    store = resolve_store()
    state = AppState(store=store, db_url="test")
    app = RapmatApp(state)

    # We just want to instantiate the screens and make sure they don't crash
    from rapmat.tui.screens.csp_search import CSPSearchScreen
    from rapmat.tui.screens.study_create import StudyCreateScreen

    print("Building StudyCreateScreen...")
    s1 = StudyCreateScreen(state, app._router)
    s1.build()
    print("StudyCreateScreen built successfully.")

    print("Building CSPSearchScreen...")
    s2 = CSPSearchScreen(state, app._router)
    s2.build()
    print("CSPSearchScreen built successfully.")


if __name__ == "__main__":
    test_tui()
