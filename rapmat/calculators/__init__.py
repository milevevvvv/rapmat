import subprocess
import importlib.util

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
    try:
        return importlib.util.find_spec(CALCULATOR_META[calc]["probe"]) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def get_install_hint(calc: Calculators) -> str | None:
    extra = CALCULATOR_META[calc]["extra"]
    return f"pip install rapmat[{extra}]" if extra else None


# ------------------------------------------------------------------ #
#  Calculator loading callback protocol
# ------------------------------------------------------------------ #


@runtime_checkable
class CalculatorCallback(Protocol):
    def on_status(self, message: str) -> None: ...


def _notify(callback: CalculatorCallback | None, message: str) -> None:
    if callback is not None:
        callback.on_status(message)


def ensure_asset(
    name: str,
    path: Path,
    install_fn: Callable[[], subprocess.CompletedProcess],
    callback: CalculatorCallback | None = None,
    log_path: Path | None = None,
) -> None:
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


def cleanup_calculator_files(calculator) -> None:
    calc_name = getattr(calculator, "name", "").lower()

    if calc_name == "vasp" and hasattr(calculator, "directory"):
        for fname in [
            "WAVECAR",
            "WAVECAR.h5",
            "CHGCAR",
            "CHG",
            "vasprun.xml",
            "OUTCAR",
            "OSZICAR",
            "EIGENVAL",
            "DOSCAR",
            "PROCAR",
            "IBZKPT",
            "PCDAT",
            "XDATCAR",
            "CONTCAR",
            "vasp.out",
            "vasp*.lock",
        ]:
            fpath = Path(calculator.directory) / fname
            if fpath.exists():
                try:
                    fpath.unlink()
                except OSError:
                    pass
