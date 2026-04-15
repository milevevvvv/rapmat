import numpy as np

from ase import Atoms
from ase.neighborlist import neighbor_list


def min_interatomic_distance(atoms: Atoms, cutoff: float = 5.0) -> float:
    if len(atoms) < 2:
        return float("inf")
    i, j, d = neighbor_list("ijd", atoms, cutoff=cutoff)
    if len(d) == 0:
        return float("inf")
    return float(np.min(d))


def has_close_contacts_pymatgen(atoms: Atoms, tolerance: float = 0.5) -> bool:
    from pymatgen.io.ase import AseAtomsAdaptor

    struct = AseAtomsAdaptor.get_structure(atoms)
    return not struct.is_valid(tol=tolerance)


def check_sanity(
    atoms: Atoms,
    min_dist: float = 0.5,
    use_pymatgen: bool = False,
    pymatgen_tol: float = 0.5,
) -> bool:
    dmin = min_interatomic_distance(atoms)
    if dmin < min_dist:
        return False
    if use_pymatgen:
        if has_close_contacts_pymatgen(atoms, tolerance=pymatgen_tol):
            return False
    return True
