import json
import urwid


def build_config_grid(config: dict, cell_width: int = 35) -> urwid.Widget:
    if not config:
        return urwid.Text([("details", "No configuration parameters available.")])

    cells = []
    for k in sorted(config.keys()):
        val = config[k]
        if isinstance(val, dict):
            try:
                val_str = json.dumps(val)
            except Exception:
                val_str = str(val)
        else:
            val_str = str(val)

        label = str(k).replace("_", " ").title() + ":"
        markup = [("form_label", f"{label} "), ("details", val_str)]
        cells.append(urwid.Text(markup, align="left"))

    return urwid.GridFlow(cells, cell_width=cell_width, h_sep=2, v_sep=1, align="left")
