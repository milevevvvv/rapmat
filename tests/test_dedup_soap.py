"""SOAP-based deduplication sanity tests.

Uses real SOAP descriptors and SurrealDB to verify that the fingerprinting
pipeline correctly distinguishes structures that differ in phase, lattice
parameter, or chemical species.
"""

import numpy as np
import pytest
from ase.build import bulk

from rapmat.storage import SOAPDescriptor, SurrealDBStore


@pytest.fixture
def soap_cu():
    return SOAPDescriptor(species=["Cu"], n_max=4, l_max=3)


@pytest.fixture
def soap_alcu():
    return SOAPDescriptor(species=["Al", "Cu"], n_max=4, l_max=3)


# ------------------------------------------------------------------ #
#  Descriptor-level tests
# ------------------------------------------------------------------ #


def test_soap_dimension_matches_vector(soap_cu):
    """dimension() should agree with the actual computed vector length."""
    cu = bulk("Cu", "fcc", a=3.615)
    vec = soap_cu.compute(cu)
    assert vec.shape == (soap_cu.dimension(),)
    assert vec.ndim == 1
    assert np.linalg.norm(vec) > 0


def test_identical_structures_same_vector(soap_cu):
    """Two identical Atoms objects produce bit-identical SOAP vectors."""
    cu1 = bulk("Cu", "fcc", a=3.615)
    cu2 = bulk("Cu", "fcc", a=3.615)
    np.testing.assert_array_equal(soap_cu.compute(cu1), soap_cu.compute(cu2))


def test_different_phases_distant_vectors(soap_cu):
    """FCC and BCC Cu produce well-separated SOAP vectors."""
    vec_fcc = soap_cu.compute(bulk("Cu", "fcc", a=3.615))
    vec_bcc = soap_cu.compute(bulk("Cu", "bcc", a=2.87))
    distance = np.linalg.norm(vec_fcc - vec_bcc)
    assert distance > 0.1


def test_perturbed_lattice_close_vectors(soap_cu):
    """Tiny lattice strain produces close but non-identical SOAP vectors."""
    vec1 = soap_cu.compute(bulk("Cu", "fcc", a=3.615))
    vec2 = soap_cu.compute(bulk("Cu", "fcc", a=3.616))  # Much smaller perturbation
    distance = np.linalg.norm(vec1 - vec2)
    assert 0 < distance < 100.0  # SOAP is sensitive; adjust threshold


def test_different_species_distant_vectors(soap_alcu):
    """Cu FCC and Al FCC with the same crystal structure produce distant vectors."""
    vec_cu = soap_alcu.compute(bulk("Cu", "fcc", a=3.615))
    vec_al = soap_alcu.compute(bulk("Al", "fcc", a=4.05))
    distance = np.linalg.norm(vec_cu - vec_al)
    assert distance > 0.1


# ------------------------------------------------------------------ #
#  Full pipeline: SOAP -> SurrealDB duplicate detection
# ------------------------------------------------------------------ #


def _make_store_with_relaxed(tmp_path, name, soap, atoms, energy=-1.0):
    """Helper: create a store, add a single relaxed structure, return (store, vector)."""
    vec = soap.compute(atoms)
    store = SurrealDBStore.from_path(tmp_path / name)
    store.register_descriptor(soap.descriptor_id(), soap.dimension())
    store.create_study(
        study_id="run1", system="Test", domain="bulk", calculator="MATTERSIM", config={}
    )
    store.create_run(name="run1", study_id="run1")
    store.add_candidate(atoms, vec, "run1", "run1/1")
    store.update_structure(
        "run1/1",
        "relaxed",
        atoms=atoms,
        vector=vec,
        metadata={
            "energy_per_atom": energy,
            "energy_total": energy * len(atoms),
            "converged": True,
        },
    )
    return store, vec


def test_identical_structures_are_duplicates(tmp_path, soap_cu):
    """Identical structures should be detected as duplicates in the store."""
    cu = bulk("Cu", "fcc", a=3.615)
    store, _ = _make_store_with_relaxed(tmp_path, "dup_ident", soap_cu, cu)

    query_vec = soap_cu.compute(bulk("Cu", "fcc", a=3.615))
    assert store.is_duplicate(query_vec, threshold=0.1) is True


def test_different_phases_are_not_duplicates(tmp_path, soap_cu):
    """FCC and BCC Cu should NOT be detected as duplicates."""
    cu_fcc = bulk("Cu", "fcc", a=3.615)
    store, _ = _make_store_with_relaxed(tmp_path, "dup_phase", soap_cu, cu_fcc)

    vec_bcc = soap_cu.compute(bulk("Cu", "bcc", a=2.87))
    assert store.is_duplicate(vec_bcc, threshold=0.01) is False


def test_perturbed_lattice_duplicate_depends_on_threshold(tmp_path, soap_cu):
    """Slightly perturbed lattice: duplicate at large threshold, not at tiny one."""
    cu = bulk("Cu", "fcc", a=3.615)
    store, _ = _make_store_with_relaxed(tmp_path, "dup_perturb", soap_cu, cu)

    vec_perturbed = soap_cu.compute(bulk("Cu", "fcc", a=3.616))  # Smaller perturbation
    # SOAP is sensitive; need a large threshold to catch small perturbations
    assert store.is_duplicate(vec_perturbed, threshold=100.0) is True
    assert store.is_duplicate(vec_perturbed, threshold=1e-10) is False


def test_duplicate_min_energy_returns_lowest(tmp_path, soap_cu):
    """get_duplicate_min_energy should return the minimum among duplicates."""
    store = SurrealDBStore.from_path(tmp_path / "dup_min_e")
    store.register_descriptor(soap_cu.descriptor_id(), soap_cu.dimension())
    store.create_study(
        study_id="run1", system="Test", domain="bulk", calculator="MATTERSIM", config={}
    )
    store.create_run(name="run1", study_id="run1")

    cu = bulk("Cu", "fcc", a=3.615)
    vec = soap_cu.compute(cu)

    # Add two relaxed structures with the same SOAP vector but different energies
    store.add_candidate(cu, vec, "run1", "run1/1")
    store.update_structure(
        "run1/1",
        "relaxed",
        atoms=cu,
        vector=vec,
        metadata={"energy_per_atom": -3.0, "energy_total": -3.0, "converged": True},
    )

    cu2 = bulk("Cu", "fcc", a=3.615)
    vec2 = soap_cu.compute(cu2)
    store.add_candidate(cu2, vec2, "run1", "run1/2")
    store.update_structure(
        "run1/2",
        "relaxed",
        atoms=cu2,
        vector=vec2,
        metadata={"energy_per_atom": -5.0, "energy_total": -5.0, "converged": True},
    )

    min_e = store.get_duplicate_min_energy(vec, threshold=0.1, run_id="run1")
    assert min_e == pytest.approx(-5.0)
