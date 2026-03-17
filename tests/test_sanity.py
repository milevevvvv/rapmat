"""Tests for physical sanity checks on relaxed structures."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from rapmat.core.sanity import (
    check_sanity,
    has_close_contacts_pymatgen,
    min_interatomic_distance,
)


class TestMinInteratomicDistance:
    def test_si_diamond(self):
        atoms = bulk("Si", "diamond", a=5.43)
        d = min_interatomic_distance(atoms)
        # Si diamond: nearest-neighbor distance = a * sqrt(3) / 4 ≈ 2.35 A
        assert d == pytest.approx(2.35, rel=0.01)

    def test_collapsed_structure(self):
        atoms = Atoms(
            "Si2",
            positions=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        d = min_interatomic_distance(atoms)
        assert d == pytest.approx(0.1, rel=0.01)

    def test_single_atom_returns_inf(self):
        atoms = Atoms("Si", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
        d = min_interatomic_distance(atoms)
        assert d == float("inf")

    def test_empty_structure_returns_inf(self):
        atoms = Atoms()
        d = min_interatomic_distance(atoms)
        assert d == float("inf")


class TestCheckSanity:
    def test_normal_structure_passes(self):
        atoms = bulk("Si", "diamond", a=5.43)
        assert check_sanity(atoms) is True

    def test_collapsed_structure_fails(self):
        atoms = Atoms(
            "Si2",
            positions=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        assert check_sanity(atoms, min_dist=0.5) is False

    def test_collapsed_with_lax_threshold_passes(self):
        atoms = Atoms(
            "Si2",
            positions=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        assert check_sanity(atoms, min_dist=0.05) is True

    def test_zno_passes(self):
        atoms = bulk("ZnO", "wurtzite", a=3.25, c=5.2)
        assert check_sanity(atoms) is True


class TestHasCloseContactsPymatgen:
    @pytest.fixture(autouse=True)
    def skip_if_no_pymatgen(self):
        try:
            from pymatgen.core import Structure
        except ImportError:
            pytest.skip("pymatgen not available")

    def test_normal_structure_no_close_contacts(self):
        atoms = bulk("Si", "diamond", a=5.43)
        assert has_close_contacts_pymatgen(atoms, tolerance=0.5) is False

    def test_collapsed_structure_has_close_contacts(self):
        atoms = Atoms(
            "Si2",
            positions=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        assert has_close_contacts_pymatgen(atoms, tolerance=0.5) is True


class TestCheckSanityWithPymatgen:
    @pytest.fixture(autouse=True)
    def skip_if_no_pymatgen(self):
        try:
            from pymatgen.core import Structure
        except ImportError:
            pytest.skip("pymatgen not available")

    def test_normal_structure_passes_with_pymatgen(self):
        atoms = bulk("Si", "diamond", a=5.43)
        assert check_sanity(atoms, use_pymatgen=True) is True

    def test_collapsed_structure_fails_with_pymatgen(self):
        atoms = Atoms(
            "Si2",
            positions=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        assert check_sanity(atoms, min_dist=0.5, use_pymatgen=True) is False
