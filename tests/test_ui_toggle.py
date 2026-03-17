import urwid
from rapmat.tui.widgets.form import FormGroup, checkbox_field, text_field

def test():
    fields = [
        checkbox_field("grid_search", "Compositional Grid Search", default=False),
        text_field("formula", "Formula"),
    ]
    form = FormGroup(fields)

    def _sync(widget=None, state=None):
        vals = form.get_values()
        is_grid = vals.get("grid_search", False)
        # Simply disable formula if grid is true
        form.set_field_disabled("formula", is_grid)
        
        # log to a file so we can see
        with open("ui_log.txt", "a") as f:
            f.write(f"Sync called. Grid from vals={is_grid}, state from callback={state}\n")

    grid_cb = form.get_widget("grid_search")
    urwid.connect_signal(grid_cb, "change", _sync)
    
    # Just render it or test clicking it programmatically
    # Simulating click:
    grid_cb.keypress((10,), " ")
    grid_cb.keypress((10,), " ")
    grid_cb.keypress((10,), " ")
    grid_cb.keypress((10,), " ")

if __name__ == "__main__":
    with open("ui_log.txt", "w") as f:
        f.write("")
    test()
