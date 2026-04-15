import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Tuple

import chemparse
from ase.data import atomic_numbers

from rapmat.config import APP_TMPDIR_SUFFIX


def parse_formula(formula: str) -> dict[str, int]:
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
    if min(formula_units) < 1:
        raise ValueError("The number of formula units can't be lower than 1.")

    if formula_units[0] > formula_units[1]:
        raise ValueError("The lower bound can't be higher than the upper one.")


def parse_system(system: str) -> list[str]:
    elements = [e.strip() for e in system.split("-") if e.strip()]
    if not elements:
        raise ValueError(f"Invalid system string: '{system}'.")

    for e in elements:
        if e not in atomic_numbers:
            raise ValueError(f"Invalid element symbol: '{e}' in system '{system}'.")

    return sorted(set(elements))


def format_system(elements: list[str]) -> str:
    return "-".join(sorted(set(elements)))


@contextmanager
def workdir_context(workdir: str | None) -> Generator[Path, None, None]:
    if workdir is None:
        with tempfile.TemporaryDirectory(suffix=APP_TMPDIR_SUFFIX) as td:
            yield Path(td)
    else:
        path = Path(workdir)
        path.mkdir(parents=True, exist_ok=True)
        yield path.resolve()
