import sys
import urwid

def test_checkbox():
    cb = urwid.CheckBox("Grid Search", state=False)
    
    def on_change(widget, state):
        print(f"Signal callback: widget.get_state()={widget.get_state()}, state arg={state}")

    urwid.connect_signal(cb, "change", on_change)
    
    print("Initial state:", cb.get_state())
    print("Toggling to True...")
    cb.set_state(True)
    print("Toggling to False...")
    cb.set_state(False)

if __name__ == "__main__":
    test_checkbox()
