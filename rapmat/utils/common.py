import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Tuple

from ase.data import atomic_numbers
import chemparse
from rapmat.config import APP_TMPDIR_SUFFIX


def parse_formula(formula: str) -> dict[str, int]:
    """Convert a chemical formula string like ``Al2O3`` into ``{"Al": 2, "O": 3}``.

    Uses chemparse for parsing; supports parentheses and brackets (e.g. K4[Fe(SCN)6]).
    Only integer stoichiometries are allowed; fractional counts raise ValueError.
    """
    raw = chemparse.parse_formula(formula)
    counts: dict[str, int] = {}
    for elem, val in raw.items():
        if val != int(val) or val < 1:
            raise ValueError(
                f"Formula must have integer stoichiometry; got {elem}{val} in '{formula}'."
            )
        counts[elem] = int(val)
    return counts


def validate_formula_units(formula_units: Tuple[int, int]) -> None:
    """Raise if formula-unit bounds are invalid."""
    if min(formula_units) < 1:
        raise ValueError("The number of formula units can't be lower than 1.")

    if formula_units[0] > formula_units[1]:
        raise ValueError("The lower bound can't be higher than the upper one.")


def parse_system(system: str) -> list[str]:
    """Parse a chemical system string like ``"Al-O"`` into ``["Al", "O"]``.

    Each component is verified to be a valid chemical element symbol.
    Elements are returned sorted alphabetically.
    """
    elements = [e.strip() for e in system.split("-") if e.strip()]
    if not elements:
        raise ValueError(f"Invalid system string: '{system}'.")

    for e in elements:
        if e not in atomic_numbers:
            raise ValueError(f"Invalid element symbol: '{e}' in system '{system}'.")

    return sorted(set(elements))


def format_system(elements: list[str]) -> str:
    """Format a list of element symbols into a canonical system string.

    Elements are sorted alphabetically and joined with ``"-"``.
    """
    return "-".join(sorted(set(elements)))


@contextmanager
def workdir_context(workdir: str | None) -> Generator[Path, None, None]:
    """Yield a resolved workdir ``Path``, creating a temp directory when *workdir* is ``None``."""
    if workdir is None:
        with tempfile.TemporaryDirectory(suffix=APP_TMPDIR_SUFFIX) as td:
            yield Path(td)
    else:
        path = Path(workdir)
        path.mkdir(parents=True, exist_ok=True)
        yield path.resolve()
