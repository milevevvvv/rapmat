import sys
import urwid


def test_render():
    from rapmat.db_config import resolve_store
    from rapmat.tui.app import RapmatApp
    from rapmat.tui.state import AppState
    from rapmat.tui.screens.home import HomeScreen

    store = resolve_store()
    state = AppState(store=store, db_url="test")
    app = RapmatApp(state)

    screen = HomeScreen(state, app._router)
    widget = screen.build()

    # The crash occurred at 121x13
    size = (121, 13)
    try:
        # We need to simulate the frame body rendering
        canv = widget.render(size, focus=True)
        print("Success! Canvas size:", canv.cols(), "x", canv.rows())
    except Exception as e:
        print("Crash:", e)
        sys.exit(1)


if __name__ == "__main__":
    test_render()
