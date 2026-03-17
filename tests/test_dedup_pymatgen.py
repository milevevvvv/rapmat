"""Tests for pymatgen StructureMatcher deduplication (stage 2).

Verifies that ``confirm_duplicates`` correctly uses pymatgen to distinguish
true crystallographic duplicates from descriptor collisions.
"""

import numpy as np
import pytest
from ase.build import bulk

from rapmat.core.dedup import confirm_duplicates
from rapmat.storage import SOAPDescriptor, SurrealDBStore

# ------------------------------------------------------------------ #
#  Unit tests for confirm_duplicates()
# ------------------------------------------------------------------ #


def _nearby(atoms, energy_per_atom, distance=0.001, struct_id="x/1"):
    """Build a dict matching the get_nearby_relaxed_structures() format."""
    return {
        "id": struct_id,
        "atoms": atoms,
        "energy_per_atom": energy_per_atom,
        "distance": distance,
    }


class TestConfirmDuplicatesWithoutPymatgen:
    """When use_pymatgen=False, all nearby entries count as duplicates."""

    def test_empty_nearby_returns_none(self):
        cu = bulk("Cu", "fcc", a=3.615)
        assert confirm_duplicates(cu, []) is None

    def test_returns_min_energy(self):
        cu = bulk("Cu", "fcc", a=3.615)
        nearby = [
            _nearby(cu, -3.0),
            _nearby(cu, -5.0, struct_id="x/2"),
        ]
        assert confirm_duplicates(cu, nearby, use_pymatgen=False) == pytest.approx(-5.0)

    def test_single_entry(self):
        cu = bulk("Cu", "fcc", a=3.615)
        nearby = [_nearby(cu, -2.5)]
        assert confirm_duplicates(cu, nearby, use_pymatgen=False) == pytest.approx(-2.5)


class TestConfirmDuplicatesWithPymatgen:
    """When use_pymatgen=True, only crystallographic matches count."""

    def test_same_structure_confirmed(self):
        cu = bulk("Cu", "fcc", a=3.615)
        cu_dup = bulk("Cu", "fcc", a=3.615)
        nearby = [_nearby(cu_dup, -3.5)]
        result = confirm_duplicates(cu, nearby, use_pymatgen=True)
        assert result == pytest.approx(-3.5)

    def test_different_phase_rejected(self):
        """FCC vs BCC should NOT be confirmed as duplicates."""
        cu_fcc = bulk("Cu", "fcc", a=3.615)
        cu_bcc = bulk("Cu", "bcc", a=2.87)
        nearby = [_nearby(cu_bcc, -4.0)]
        result = confirm_duplicates(cu_fcc, nearby, use_pymatgen=True)
        assert result is None

    def test_slightly_perturbed_confirmed(self):
        """Small lattice perturbation should still match with default tolerances."""
        cu = bulk("Cu", "fcc", a=3.615)
        cu_perturbed = bulk("Cu", "fcc", a=3.620)
        nearby = [_nearby(cu_perturbed, -3.0)]
        result = confirm_duplicates(cu, nearby, use_pymatgen=True)
        assert result == pytest.approx(-3.0)

    def test_mixed_nearby_filters_collisions(self):
        """Only crystallographic matches contribute to min energy."""
        cu_fcc = bulk("Cu", "fcc", a=3.615)
        cu_bcc = bulk("Cu", "bcc", a=2.87)
        cu_fcc_dup = bulk("Cu", "fcc", a=3.615)

        nearby = [
            _nearby(cu_bcc, -10.0, struct_id="x/1"),
            _nearby(cu_fcc_dup, -3.0, struct_id="x/2"),
        ]
        result = confirm_duplicates(cu_fcc, nearby, use_pymatgen=True)
        assert result == pytest.approx(-3.0)

    def test_all_rejected_returns_none(self):
        """If pymatgen rejects all nearby, return None (keep the structure)."""
        cu_fcc = bulk("Cu", "fcc", a=3.615)
        cu_bcc = bulk("Cu", "bcc", a=2.87)
        al_fcc = bulk("Al", "fcc", a=4.05)

        nearby = [
            _nearby(cu_bcc, -5.0, struct_id="x/1"),
            _nearby(al_fcc, -3.0, struct_id="x/2"),
        ]
        result = confirm_duplicates(cu_fcc, nearby, use_pymatgen=True)
        assert result is None


class TestPymatgenTolerances:
    """Tolerance parameters affect StructureMatcher sensitivity."""

    def test_tight_tolerances_reject_perturbation(self):
        """Large internal-coordinate shift in a 2-atom cell rejected by tight stol."""
        si = bulk("Si", "diamond", a=5.43)
        si_off = si.copy()
        si_off.positions[1] += [0.5, 0.5, 0.0]
        nearby = [_nearby(si_off, -2.0)]
        result = confirm_duplicates(
            si,
            nearby,
            use_pymatgen=True,
            ltol=0.01,
            stol=0.01,
            angle_tol=0.1,
        )
        assert result is None

    def test_loose_tolerances_accept_perturbation(self):
        """Same displacement accepted by loose stol."""
        si = bulk("Si", "diamond", a=5.43)
        si_off = si.copy()
        si_off.positions[1] += [0.5, 0.5, 0.0]
        nearby = [_nearby(si_off, -2.0)]
        result = confirm_duplicates(
            si,
            nearby,
            use_pymatgen=True,
            ltol=0.5,
            stol=0.5,
            angle_tol=10.0,
        )
        assert result == pytest.approx(-2.0)


# ------------------------------------------------------------------ #
#  Integration: store.get_nearby_relaxed_structures + confirm_duplicates
# ------------------------------------------------------------------ #


class TestStoreIntegration:
    """End-to-end with SurrealDB store and real SOAP descriptors."""

    @pytest.fixture
    def soap(self):
        return SOAPDescriptor(species=["Cu"], n_max=4, l_max=3)

    def _make_store(self, tmp_path, name, soap, atoms, energy):
        store = SurrealDBStore.from_path(tmp_path / name)
        store.register_descriptor(soap.descriptor_id(), soap.dimension())
        store.create_study(study_id="run1", system="Test", domain="bulk", calculator="MATTERSIM", config={})
        store.create_run(name="run1", study_id="run1")
        vec = soap.compute(atoms)
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

    def test_nearby_returns_atoms(self, tmp_path, soap):
        cu = bulk("Cu", "fcc", a=3.615)
        store, vec = self._make_store(tmp_path, "nearby_atoms", soap, cu, -3.0)

        nearby = store.get_nearby_relaxed_structures(vec, threshold=1.0, run_id="run1")
        assert len(nearby) == 1
        assert nearby[0]["energy_per_atom"] == pytest.approx(-3.0)
        assert nearby[0]["atoms"] is not None
        assert len(nearby[0]["atoms"]) == len(cu)

    def test_nearby_empty_when_distant(self, tmp_path, soap):
        cu_fcc = bulk("Cu", "fcc", a=3.615)
        store, _ = self._make_store(tmp_path, "nearby_dist", soap, cu_fcc, -3.0)

        cu_bcc = bulk("Cu", "bcc", a=2.87)
        vec_bcc = soap.compute(cu_bcc)
        nearby = store.get_nearby_relaxed_structures(
            vec_bcc, threshold=1e-5, run_id="run1"
        )
        assert len(nearby) == 0

    def test_full_pipeline_pymatgen_catches_collision(self, tmp_path, soap):
        """Simulate a collision: BCC stored, FCC query has close vector.

        With pymatgen enabled the collision is caught and the query is
        *not* treated as a duplicate.
        """
        cu_bcc = bulk("Cu", "bcc", a=2.87)
        store, _ = self._make_store(tmp_path, "collision", soap, cu_bcc, -4.0)

        cu_fcc = bulk("Cu", "fcc", a=3.615)
        vec_fcc = soap.compute(cu_fcc)

        nearby = store.get_nearby_relaxed_structures(
            vec_fcc, threshold=999.0, run_id="run1"
        )
        assert len(nearby) >= 1

        result = confirm_duplicates(cu_fcc, nearby, use_pymatgen=True)
        assert result is None

    def test_get_nearby_structures_multiple_statuses(self, tmp_path, soap):
        """get_nearby_structures finds both generated and relaxed entries."""
        store = SurrealDBStore.from_path(tmp_path / "multi_status")
        store.register_descriptor(soap.descriptor_id(), soap.dimension())
        store.create_study(study_id="run1", system="Test", domain="bulk", calculator="MATTERSIM", config={})
        store.create_run(name="run1", study_id="run1")

        cu_fcc = bulk("Cu", "fcc", a=3.615)
        vec = soap.compute(cu_fcc)

        # One relaxed structure
        store.add_candidate(cu_fcc, vec, "run1", "run1/1")
        store.update_structure(
            "run1/1",
            "relaxed",
            atoms=cu_fcc,
            vector=vec,
            metadata={
                "energy_per_atom": -3.0,
                "energy_total": -3.0,
                "converged": True,
            },
        )

        # One generated structure (with vector)
        cu_fcc2 = bulk("Cu", "fcc", a=3.62)
        vec2 = soap.compute(cu_fcc2)
        store.add_candidate(cu_fcc2, vec2, "run1", "run1/2")

        # Relaxed-only query should return 1
        relaxed_only = store.get_nearby_structures(
            vec, threshold=999.0, run_id="run1", statuses=("relaxed",)
        )
        assert len(relaxed_only) == 1
        assert relaxed_only[0]["id"] == "run1/1"

        # Multi-status query should return 2
        both = store.get_nearby_structures(
            vec, threshold=999.0, run_id="run1", statuses=("generated", "relaxed")
        )
        assert len(both) == 2
        ids = {e["id"] for e in both}
        assert "run1/1" in ids
        assert "run1/2" in ids
