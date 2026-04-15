"""Tests for structure relaxation using ASE's built-in EMT potential.

EMT is deterministic and fast, supporting Al, Cu, Ni, Pd, Ag, Pt, Au.
SinglePointCalculator is used for controlled edge-case testing.
"""

import numpy as np
import pytest
from ase.build import bulk, fcc111
from ase.calculators.emt import EMT

from rapmat.core.relaxation import structure_relax

# ------------------------------------------------------------------ #
#  Happy-path relaxation with EMT
# ------------------------------------------------------------------ #


def test_relax_converges_emt():
    """Cu FCC at off-equilibrium lattice constant should converge with EMT."""
    atoms = bulk("Cu", "fcc", a=3.7)
    atoms.calc = EMT()

    converged, relaxed = structure_relax(
        atoms,
        force_conv_crit=0.05,
        steps_max=200,
    )

    assert converged is True
    forces = relaxed.get_forces()
    fmax = float(np.max(np.linalg.norm(forces, axis=1)))
    assert fmax < 0.05


def test_relax_energy_decreases():
    """Relaxation should lower the energy from the initial strained state."""
    atoms = bulk("Al", "fcc", a=4.3)
    atoms.calc = EMT()
    e_before = atoms.get_potential_energy()

    converged, relaxed = structure_relax(
        atoms,
        force_conv_crit=0.05,
        steps_max=200,
    )

    e_after = relaxed.get_potential_energy()
    assert e_after < e_before


# ------------------------------------------------------------------ #
#  Error handling
# ------------------------------------------------------------------ #


def test_relax_no_calculator_raises():
    """Structure without a calculator should raise RuntimeError."""
    atoms = bulk("Cu", "fcc", a=3.615)
    with pytest.raises(RuntimeError, match="No calculator"):
        structure_relax(atoms)


def test_relax_force_break_aborts():
    """Relaxation aborts early when forces exceed forces_break.

    Use SinglePointCalculator to inject huge forces that exceed forces_break.
    FrechetCellFilter requires stress, so we provide that too.
    """
    from ase.calculators.singlepoint import SinglePointCalculator

    atoms = bulk("Cu", "fcc", a=3.615)
    # Inject huge forces that exceed forces_break
    huge_forces = np.ones((len(atoms), 3)) * 1e7  # 1e7 eV/A per component
    stress = np.zeros(6)  # Zero stress for simplicity
    atoms.calc = SinglePointCalculator(
        atoms, energy=-10.0, forces=huge_forces, stress=stress
    )

    converged, _ = structure_relax(
        atoms,
        forces_break=1e6,  # Threshold below the injected forces
        steps_max=200,
    )

    assert converged is False


# ------------------------------------------------------------------ #
#  Cell filter mask
# ------------------------------------------------------------------ #


def test_relax_mask_preserves_z_cell():
    """With mask=[1,1,0,0,0,1] the z-cell vector should remain unchanged."""
    slab = fcc111("Al", size=(2, 2, 2), vacuum=10.0)
    slab.calc = EMT()
    c_before = float(slab.cell[2, 2])

    _, relaxed = structure_relax(
        slab,
        mask=[1, 1, 0, 0, 0, 1],
        force_conv_crit=0.05,
        steps_max=50,
    )

    c_after = float(relaxed.cell[2, 2])
    assert c_after == pytest.approx(c_before, abs=1e-10)
