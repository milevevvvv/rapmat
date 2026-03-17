from rich.console import Console

console = Console()
err_console = Console(stderr=True)


def silence() -> None:
    """Silence all global rich consoles, useful for TUI applications."""
    global console, err_console
    console.quiet = True
    err_console.quiet = True
