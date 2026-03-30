import importlib.util
import subprocess
from enum import Enum
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable


class Calculators(str, Enum):
    MATTERSIM = "MATTERSIM"
    NEQUIP_OAML = "NEQUIP-OAML"
    UPET = "UPET"
    VASP = "VASP"


CALCULATOR_META: dict[Calculators, dict] = {
    Calculators.MATTERSIM: {
        "probe": "mattersim.forcefield",
        "extra": "mattersim",
        "description": "MatterSim universal MLIP",
    },
    Calculators.NEQUIP_OAML: {
        "probe": "nequip.ase",
        "extra": "nequip",
        "description": "NequIP OAM-L equivariant potential",
    },
    Calculators.UPET: {
        "probe": "upet.calculator",
        "extra": "upet",
        "description": "UPET universal PET-based potential",
    },
    Calculators.VASP: {
        "probe": "ase.calculators.vasp",
        "extra": None,
        "description": "VASP (system install, no pip extra)",
    },
}


def is_calculator_available(calc: Calculators) -> bool:
    """Check whether the required package for *calc* is importable."""
    try:
        return importlib.util.find_spec(CALCULATOR_META[calc]["probe"]) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def get_install_hint(calc: Calculators) -> str | None:
    """Return a pip install command for *calc*, or None for system-level calculators."""
    extra = CALCULATOR_META[calc]["extra"]
    return f"pip install rapmat[{extra}]" if extra else None


# ------------------------------------------------------------------ #
#  Calculator loading callback protocol
# ------------------------------------------------------------------ #


@runtime_checkable
class CalculatorCallback(Protocol):
    """Optional status reporting during calculator loading / asset installation."""

    def on_status(self, message: str) -> None: ...


def _notify(callback: CalculatorCallback | None, message: str) -> None:
    """Helper: call *callback.on_status* if a callback is provided."""
    if callback is not None:
        callback.on_status(message)


def ensure_asset(
    name: str,
    path: Path,
    install_fn: Callable[[], subprocess.CompletedProcess],
    callback: CalculatorCallback | None = None,
    log_path: Path | None = None,
) -> None:
    """Download / compile an asset if it is missing.

    Parameters
    ----------
    name:
        Human-readable asset name (e.g. "NequIP-OAM-L checkpoint").
    path:
        Expected file path after a successful install.
    install_fn:
        A callable that performs the install and returns a
        ``subprocess.CompletedProcess``.  It should use ``check=False``.
    callback:
        Optional callback for progress reporting.

    Raises
    ------
    RuntimeError
        If the install command fails or the file still does not exist
        after a successful return code.
    """
    if path.exists():
        return

    _notify(callback, f"{name} not found at {path}, installing...")

    try:
        result = install_fn()
    except Exception as e:
        raise RuntimeError(f"Failed to install {name}: {e}") from e

    if result.returncode != 0:
        stderr = result.stderr.decode() if result.stderr else ""
        log_msg = f"\nSee log file for details: {log_path}" if log_path else ""
        err_msg = stderr or "No stderr output available."
        raise RuntimeError(
            f"Failed to install {name} (exit code {result.returncode}):\n{err_msg}{log_msg}"
        )

    if not path.exists():
        raise RuntimeError(
            f"Install command succeeded but {path} still does not exist."
        )

    _notify(callback, f"{name} installed successfully.")
