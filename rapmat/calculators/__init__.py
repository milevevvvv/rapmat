import importlib.util
from enum import Enum


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
