import numpy as np
import spglib
import warnings

from ase import Atoms
from typing import Optional, Tuple


def _spglib_cell(atoms: Atoms) -> tuple:
    return (
        atoms.get_cell().array,
        atoms.get_scaled_positions(wrap=False),
        atoms.get_atomic_numbers(),
    )


def calculate_thickness(atoms: Atoms, axis: int = 2) -> float:
    if len(atoms) == 0:
        return 0.0

    positions = atoms.get_scaled_positions(wrap=True)
    z_coords = positions[:, axis]

    z_coords = z_coords % 1.0

    if len(z_coords) == 1:
        return 0.0

    z_sorted = np.sort(z_coords)

    gaps = np.diff(z_sorted)

    periodic_gap = (z_sorted[0] + 1.0) - z_sorted[-1]
    all_gaps = np.append(gaps, periodic_gap)

    max_gap = np.max(all_gaps)
    thickness_scaled = 1.0 - max_gap

    c_length = atoms.cell.lengths()[axis]
    return thickness_scaled * c_length


def get_spacegroup_info(atoms: Atoms, symprec: float = 1e-3) -> Tuple[str, int]:
    cell = _spglib_cell(atoms)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    except Exception as e:
        raise RuntimeError(f"Spglib failed to generate a dataset: {e}") from e

    symbol = str(dataset.international)
    number = int(dataset.number)
    return symbol, number


def standardize_atoms(
    atoms: Atoms,
    symprec: float = 1e-3,
    to_primitive: bool = False,
    no_idealize: bool = False,
) -> Atoms:
    cell = _spglib_cell(atoms)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = spglib.standardize_cell(
            cell, to_primitive=to_primitive, no_idealize=no_idealize, symprec=symprec
        )
    if result is None:
        return atoms.copy()

    lattice, positions, numbers = result

    new_atoms = Atoms(
        numbers=numbers,
        scaled_positions=positions,
        cell=lattice,
        pbc=True,
    )
    new_atoms.info = atoms.info.copy()
    return new_atoms


def format_spg(atoms: Optional[Atoms], symprec: float = 1e-3) -> str:
    if atoms is None:
        return ""
    try:
        sym, num = get_spacegroup_info(atoms, symprec=symprec)
        return f"{sym} ({num})"
    except (RuntimeError, Exception):
        return "N/A"
