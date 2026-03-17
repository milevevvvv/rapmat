"""Physical sanity checks for relaxed structures."""

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list


def min_interatomic_distance(atoms: Atoms, cutoff: float = 5.0) -> float:
    """Return the minimum pairwise interatomic distance in Angstrom.

    Uses the ASE neighbor list for O(N) performance. Only pairs within
    *cutoff* are considered; for most solids the shortest bond is well
    below 5 A.

    Args:
        atoms: ASE Atoms object (periodic or non-periodic).
        cutoff: Search radius in Angstrom. Pairs beyond this are ignored.

    Returns:
        Minimum distance in Angstrom, or infinity if no pairs within cutoff.
    """
    if len(atoms) < 2:
        return float("inf")
    i, j, d = neighbor_list("ijd", atoms, cutoff=cutoff)
    if len(d) == 0:
        return float("inf")
    return float(np.min(d))


def has_close_contacts_pymatgen(atoms: Atoms, tolerance: float = 0.5) -> bool:
    """Check for unphysical close contacts via pymatgen Structure.is_valid.

    Uses pymatgen's built-in validation: returns True (has close contacts)
    when Structure.is_valid(tol=tolerance) is False. The tol parameter
    is a distance tolerance in Angstrom; default 0.5 flags atoms closer
    than 0.5 A.

    Args:
        atoms: ASE Atoms object.
        tolerance: Distance tolerance in Angstrom (passed to is_valid).

    Returns:
        True if the structure has close contacts (unphysical).
    """
    from pymatgen.io.ase import AseAtomsAdaptor

    struct = AseAtomsAdaptor.get_structure(atoms)
    return not struct.is_valid(tol=tolerance)


def check_sanity(
    atoms: Atoms,
    min_dist: float = 0.5,
    use_pymatgen: bool = False,
    pymatgen_tol: float = 0.5,
) -> bool:
    """Run physical sanity checks on a relaxed structure.

    Always runs the ASE-based minimum interatomic distance check.
    Optionally runs the pymatgen covalent-radius check.

    Args:
        atoms: ASE Atoms object (typically a relaxed structure).
        min_dist: Minimum allowed distance in Angstrom. Structures
            with any pair closer than this fail.
        use_pymatgen: If True, also run the pymatgen covalent-radius check.
        pymatgen_tol: Distance tolerance in Angstrom for pymatgen check.

    Returns:
        True if the structure passes all enabled checks.
    """
    dmin = min_interatomic_distance(atoms)
    if dmin < min_dist:
        return False
    if use_pymatgen:
        if has_close_contacts_pymatgen(atoms, tolerance=pymatgen_tol):
            return False
    return True
