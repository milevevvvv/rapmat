"""Tests for convex hull construction and formation energy math.

Every test uses a synthetic Al-Cu binary system with hand-computed energies
so correctness can be verified by arithmetic alone -- no calculator needed.

Reference energies:
    Al = -3.0 eV/atom,  Cu = -4.0 eV/atom

AlCu  (2 atoms, epa=-5.0):
    e_ref = 1*(-3.0) + 1*(-4.0) = -7.0
    formation_energy = (-10.0 - (-7.0)) / 2 = -1.5 eV/atom   -> ON hull

Al3Cu (4 atoms, epa=-3.0):
    e_ref = 3*(-3.0) + 1*(-4.0) = -13.0
    formation_energy = (-12.0 - (-13.0)) / 4 = +0.25 eV/atom  -> ABOVE hull
"""

import pytest
from ase import Atoms
from ase.build import bulk
from conftest import VECTOR_DIM, add_relaxed_structure

from rapmat.core.hull import (build_phase_diagram, get_composition_fraction,
                              get_reference_energies)
from rapmat.storage import SurrealDBStore

# ------------------------------------------------------------------ #
#  get_composition_fraction
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    "formula,element,expected",
    [
        ({"Al": 2, "O": 3}, "Al", 0.4),
        ({"Al": 2, "O": 3}, "O", 0.6),
        ({"Al": 2, "O": 3}, "Si", 0.0),
        ({"Si": 1}, "Si", 1.0),
        ({}, "Al", 0.0),
    ],
)
def test_get_composition_fraction(formula, element, expected):
    assert get_composition_fraction(formula, element) == pytest.approx(expected)


# ------------------------------------------------------------------ #
#  get_reference_energies
# ------------------------------------------------------------------ #


def test_get_reference_energies(hull_store):
    refs = get_reference_energies(hull_store, "test-study")
    assert refs["Al"] == pytest.approx(-3.0)
    assert refs["Cu"] == pytest.approx(-4.0)


def test_get_reference_energies_picks_minimum(hull_store):
    """If multiple pure-element structures exist, pick the lowest epa."""
    # Add a second Al structure with worse energy
    al2 = bulk("Al", "fcc", a=4.10)
    add_relaxed_structure(hull_store, "al-run", al2, -2.5, "al-run/2")

    refs = get_reference_energies(hull_store, "test-study")
    assert refs["Al"] == pytest.approx(-3.0)


def test_get_reference_energies_missing_endpoint(tmp_path):
    """ValueError when a pure-element run is missing."""
    store = SurrealDBStore.from_path(tmp_path / "missing_ep")
    store.create_study("s", system="Al-Cu", domain="bulk", calculator="mock")

    store.create_run(
        "al-only",
        config={"formula": {"Al": 1}, "calculator": "mock"},
        study_id="s",
    )
    add_relaxed_structure(
        store, "al-only", bulk("Al", "fcc", a=4.05), -3.0, "al-only/1"
    )

    with pytest.raises(ValueError, match="pure-Cu"):
        get_reference_energies(store, "s")


# ------------------------------------------------------------------ #
#  build_phase_diagram
# ------------------------------------------------------------------ #


def test_build_phase_diagram_stable_and_unstable(hull_store):
    pd_obj, data, _ = build_phase_diagram(hull_store, "test-study")

    by_formula = {d["reduced_formula"]: d for d in data}

    # AlCu should be stable (on the hull)
    alcu = by_formula["AlCu"]
    assert alcu["is_stable"] is True
    assert alcu["energy_above_hull"] < 1e-6
    assert alcu["formation_energy"] == pytest.approx(-1.5, abs=1e-4)
    assert alcu["composition_frac"] == pytest.approx(0.5, abs=1e-4)

    # Al3Cu should be unstable (above hull)
    al3cu = by_formula["Al3Cu"]
    assert (
        al3cu["is_stable"] == False
    )  # Use == instead of is (numpy bool vs Python bool)
    assert al3cu["energy_above_hull"] > 0.1
    assert al3cu["formation_energy"] == pytest.approx(
        0.25, abs=1e-4
    )  # Updated: +0.25 eV/atom
    assert al3cu["composition_frac"] == pytest.approx(0.25, abs=1e-4)


def test_build_phase_diagram_formation_energy_signs(hull_store):
    """Pure-element endpoints should have zero formation energy."""
    _, data, _ = build_phase_diagram(hull_store, "test-study")

    by_formula = {d["reduced_formula"]: d for d in data}

    assert by_formula["Al"]["formation_energy"] == pytest.approx(0.0, abs=1e-6)
    assert by_formula["Cu"]["formation_energy"] == pytest.approx(0.0, abs=1e-6)


def test_build_phase_diagram_show_all_vs_best(hull_store):
    """show_all=True returns every structure; False keeps only the best per composition."""
    # Add a second AlCu with worse energy (epa = -4.0 vs -5.0)
    alcu2 = Atoms(
        symbols=["Al", "Cu"],
        positions=[[0, 0, 0], [1.5, 1.5, 1.5]],
        cell=[3, 3, 3],
        pbc=True,
    )
    add_relaxed_structure(hull_store, "alcu-on", alcu2, -4.0, "alcu-on/2")

    # show_all=False: only best per composition
    _, data_best, _ = build_phase_diagram(hull_store, "test-study", show_all=False)
    alcu_best = [d for d in data_best if d["reduced_formula"] == "AlCu"]
    assert len(alcu_best) == 1
    assert alcu_best[0]["energy_per_atom"] == pytest.approx(-5.0)

    # show_all=True: all structures
    _, data_all, _ = build_phase_diagram(hull_store, "test-study", show_all=True)
    alcu_all = [d for d in data_all if d["reduced_formula"] == "AlCu"]
    assert len(alcu_all) == 2


def test_build_phase_diagram_no_intermediates(tmp_path):
    """Only pure-element endpoints should raise ValueError."""
    store = SurrealDBStore.from_path(tmp_path / "no_inter")
    store.create_study("s", system="Al-Cu", domain="bulk", calculator="mock")

    store.create_run(
        "al-r",
        config={"formula": {"Al": 1}, "calculator": "mock"},
        study_id="s",
    )
    add_relaxed_structure(store, "al-r", bulk("Al", "fcc", a=4.05), -3.0, "al-r/1")

    store.create_run(
        "cu-r",
        config={"formula": {"Cu": 1}, "calculator": "mock"},
        study_id="s",
    )
    add_relaxed_structure(store, "cu-r", bulk("Cu", "fcc", a=3.615), -4.0, "cu-r/1")

    with pytest.raises(ValueError, match="intermediate"):
        build_phase_diagram(store, "s")


def test_build_phase_diagram_study_not_found(tmp_path):
    store = SurrealDBStore.from_path(tmp_path / "empty")
    with pytest.raises(ValueError, match="not found"):
        build_phase_diagram(store, "nonexistent")


def test_build_phase_diagram_data_sorted_by_composition(hull_store):
    """Returned structure_data should be sorted by composition_frac."""
    _, data, _ = build_phase_diagram(hull_store, "test-study")
    fracs = [d["composition_frac"] for d in data]
    assert fracs == sorted(fracs)
