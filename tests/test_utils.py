"""Tests for utility functions: formula parsing, system parsing, thickness, SPG."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, fcc111
from rapmat.utils.common import parse_formula, parse_system, validate_formula_units
from rapmat.utils.structure import calculate_thickness, format_spg

# ------------------------------------------------------------------ #
#  parse_formula
# ------------------------------------------------------------------ #


class TestParseFormula:
    @pytest.mark.parametrize(
        "formula,expected",
        [
            ("Al2O3", {"Al": 2, "O": 3}),
            ("Si", {"Si": 1}),
            ("H2O", {"H": 2, "O": 1}),
            ("Fe", {"Fe": 1}),
            ("NaCl", {"Na": 1, "Cl": 1}),
        ],
    )
    def test_valid(self, formula, expected):
        assert parse_formula(formula) == expected

    def test_fractional_raises(self):
        with pytest.raises(ValueError, match="integer"):
            parse_formula("Al0.5O1.5")


# ------------------------------------------------------------------ #
#  parse_system
# ------------------------------------------------------------------ #


class TestParseSystem:
    def test_basic(self):
        assert parse_system("Al-O") == ["Al", "O"]

    def test_sorted_alphabetically(self):
        assert parse_system("O-Al") == ["Al", "O"]

    def test_whitespace_stripped(self):
        assert parse_system(" Al - O ") == ["Al", "O"]

    def test_three_elements(self):
        assert parse_system("Li-Fe-O") == ["Fe", "Li", "O"]

    def test_deduplicates(self):
        assert parse_system("Al-Al-O") == ["Al", "O"]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_system("")

    def test_only_dashes_raises(self):
        with pytest.raises(ValueError):
            parse_system("--")


# ------------------------------------------------------------------ #
#  validate_formula_units
# ------------------------------------------------------------------ #


class TestValidateFormulaUnits:
    def test_valid_range(self):
        validate_formula_units((1, 4))

    def test_single_unit(self):
        validate_formula_units((2, 2))

    def test_zero_lower_raises(self):
        with pytest.raises(ValueError, match="lower than 1"):
            validate_formula_units((0, 4))

    def test_inverted_raises(self):
        with pytest.raises(ValueError, match="lower bound"):
            validate_formula_units((4, 1))


# ------------------------------------------------------------------ #
#  calculate_thickness
# ------------------------------------------------------------------ #


class TestCalculateThickness:
    def test_slab_with_vacuum(self):
        slab = fcc111("Al", size=(1, 1, 3), vacuum=10.0)
        thickness = calculate_thickness(slab, axis=2)
        # 3 layers of Al(111) with large vacuum -- thickness is a few Angstroms
        assert 1.0 < thickness < 10.0

    def test_empty_atoms(self):
        assert calculate_thickness(Atoms()) == 0.0

    def test_single_atom(self):
        single = Atoms("H", positions=[[0, 0, 5]], cell=[10, 10, 10], pbc=True)
        assert calculate_thickness(single, axis=2) == 0.0

    def test_bulk_has_zero_thickness(self):
        """Bulk structures with evenly distributed atoms have zero thickness.

        The algorithm finds the largest gap, which for bulk is the full cell
        (gap = 1.0), giving thickness = 0. This is correct: bulk has no
        "slab thickness" in the sense of a 2D material.
        """
        cu = bulk("Cu", "fcc", a=3.615)
        thickness = calculate_thickness(cu, axis=2)
        assert thickness == 0.0


# ------------------------------------------------------------------ #
#  format_spg
# ------------------------------------------------------------------ #


class TestFormatSpg:
    def test_diamond_si(self):
        si = bulk("Si", "diamond", a=5.43)
        result = format_spg(si)
        assert "Fd-3m" in result
        assert "227" in result

    def test_fcc_cu(self):
        cu = bulk("Cu", "fcc", a=3.615)
        result = format_spg(cu)
        assert "Fm-3m" in result
        assert "225" in result

    def test_none_returns_empty(self):
        assert format_spg(None) == ""
