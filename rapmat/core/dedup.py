import warnings
from typing import Optional

import numpy as np
from ase import Atoms
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor


def _to_pymatgen(atoms: Atoms):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AseAtomsAdaptor.get_structure(atoms)


def forces_cosine_similarity(f1: np.ndarray, f2: np.ndarray) -> float:
    a = f1.ravel()
    b = f2.ravel()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0 if (norm_a < 1e-12 and norm_b < 1e-12) else 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def confirm_duplicates(
    candidate: Atoms,
    nearby: list[dict],
    *,
    use_pymatgen: bool = False,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0,
    use_forces: bool = False,
    candidate_forces: np.ndarray | None = None,
    force_cosine_threshold: float = 0.95,
) -> Optional[float]:
    if not nearby:
        return None

    remaining = list(nearby)

    if use_pymatgen:
        matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
        candidate_pmg = _to_pymatgen(candidate)
        filtered = []
        for entry in remaining:
            try:
                entry_pmg = _to_pymatgen(entry["atoms"])
                if matcher.fit(candidate_pmg, entry_pmg):
                    filtered.append(entry)
            except Exception:
                continue
        remaining = filtered

    if use_forces:
        if candidate_forces is None:
            return None
        filtered = []
        for entry in remaining:
            entry_forces = entry.get("forces")
            if entry_forces is None:
                continue
            cos_sim = forces_cosine_similarity(candidate_forces, entry_forces)
            if cos_sim >= force_cosine_threshold:
                filtered.append(entry)
        remaining = filtered

    if not remaining:
        return None
    return min(e["energy_per_atom"] for e in remaining)
