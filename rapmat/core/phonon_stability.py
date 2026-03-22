"""Batch phonon dynamical-stability computation for CSP results.

Shared logic for batch evaluation of phonon stability.
"""

from typing import Callable, List, Optional, Tuple

from ase import Atoms

from rapmat.calculators import Calculators
from rapmat.calculators.factory import load_calculator
from rapmat.core.phonon import (
    get_mesh_min_frequency,
    structure_calculate_phonons,
    structure_has_imag_phonon_freq,
)
from rapmat.utils.console import console, err_console

from rapmat.storage.base import StructureStore


def compute_dynamical_stability_for_results(
    results: List[dict],
    structures: List[Atoms],
    phonon_top: int,
    phonon_cutoff: float,
    phonon_supercell: Tuple[int, int, int],
    phonon_mesh: Tuple[int, int, int],
    phonon_displacement: float,
    phonon_calculator: Calculators,
    store: Optional["StructureStore"] = None,
    calculator_config: dict | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    symprec: float = 1e-3,
    reduce_primitive: bool = True,
) -> bool:
    """Compute phonon dynamical stability for the top N converged structures.
    """
    if phonon_top < 1:
        return False
    if not results or not structures:
        return False

    target_results: List[Tuple[int, dict]] = []
    for idx, result in enumerate(results):
        if result.get("converged"):
            target_results.append((idx, result))
            if len(target_results) >= phonon_top:
                break

    if not target_results:
        return False

    calculator = load_calculator(phonon_calculator, config=calculator_config)
    updated = False
    total = len(target_results)

    def _process_one(result_index: int, result: dict) -> None:
        nonlocal updated
        structure_index = result.get("structure_index", result_index)
        try:
            structure_index = int(structure_index)
        except (TypeError, ValueError):
            structure_index = result_index

        if structure_index < 0 or structure_index >= len(structures):
            result["min_phonon_freq"] = None
            result["dynamical_stability"] = None
            return

        atoms = structures[structure_index]
        if reduce_primitive:
            from rapmat.utils.structure import standardize_atoms
            atoms = standardize_atoms(atoms, to_primitive=True, symprec=symprec)
        atoms.calc = calculator

        try:
            phonons = structure_calculate_phonons(
                atoms,
                displacement=phonon_displacement,
                supercell=phonon_supercell,
                qpoint_mesh=phonon_mesh,
            )
            min_freq = get_mesh_min_frequency(phonons)
            result["min_phonon_freq"] = min_freq
            result["dynamical_stability"] = not structure_has_imag_phonon_freq(
                phonons, threshold=phonon_cutoff
            )
            if store is not None and result.get("structure_id"):
                store.update_structure_phonon(result["structure_id"], min_freq)
            updated = True
        except Exception as e:
            err_console.print(
                f"[red]Phonon calc failed for ID {result.get('id', structure_index + 1)}: {e}[/red]"
            )
            result["min_phonon_freq"] = None
            result["dynamical_stability"] = None
            updated = True

    for i, (result_index, result) in enumerate(target_results):
        msg = f"Structure {i + 1}/{total}: {result.get('formula', '')}"
        if progress_callback is not None:
            progress_callback(i, total, msg)
        _process_one(result_index, result)

    if progress_callback is not None:
        progress_callback(total, total, "Done")

    return updated
