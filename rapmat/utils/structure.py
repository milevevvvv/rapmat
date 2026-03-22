import warnings
from typing import Optional, Tuple

import numpy as np
import spglib
from ase import Atoms

spglib.OLD_ERROR_HANDLING = False


def calculate_thickness(atoms: Atoms, axis: int = 2) -> float:
    """Calculates the geometric thickness (span) of the structure along an axis.

    For a slab with vacuum, atoms occupy only a portion of the cell along
    the given axis. This function finds that occupied portion by treating
    the problem on a periodic 1D ring: project all atoms onto fractional
    coordinates along `axis`, find the largest empty arc (= vacuum region),
    and define thickness as everything else.

    Args:
        atoms: ASE Atoms object (slab with vacuum along `axis`).
        axis: Cell axis index to measure along (0=a, 1=b, 2=c).

    Returns:
        Thickness in Ångströms. Returns 0.0 for empty or single-atom structures.
    """
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
    no_idealize: bool = False
) -> Atoms:
    """Standardize an ASE Atoms cell using spglib.
    
    If spglib cannot standardize the cell, returns a copy of the original atoms object.
    Preserves the `info` dictionary attached to the atoms.
    """
    cell = _spglib_cell(atoms)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = spglib.standardize_cell(
            cell, 
            to_primitive=to_primitive, 
            no_idealize=no_idealize, 
            symprec=symprec
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
    """Return a formatted space-group string, e.g. ``"Fd-3m (227)"``.

    Returns ``""`` when *atoms* is ``None`` (no structure available)
    and ``"N/A"`` when spglib fails to determine the space group.
    """
    if atoms is None:
        return ""
    try:
        sym, num = get_spacegroup_info(atoms, symprec=symprec)
        return f"{sym} ({num})"
    except (RuntimeError, Exception):
        return "N/A"
