"""Tests for force-direction cosine similarity deduplication (stage 3).

Verifies ``forces_cosine_similarity`` and the force gate inside
``confirm_duplicates``.
"""

import numpy as np
import pytest
from ase.build import bulk

from rapmat.core.dedup import confirm_duplicates, forces_cosine_similarity


def _nearby(atoms, energy, forces=None, struct_id="x/1", distance=0.001):
    """Build a dict matching the get_nearby_structures() format."""
    return {
        "id": struct_id,
        "atoms": atoms,
        "energy_per_atom": energy,
        "distance": distance,
        "forces": np.asarray(forces) if forces is not None else None,
    }


# ------------------------------------------------------------------ #
#  forces_cosine_similarity
# ------------------------------------------------------------------ #


class TestForcesCosine:
    def test_identical(self):
        f = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        assert forces_cosine_similarity(f, f) == pytest.approx(1.0)

    def test_opposite(self):
        f = np.array([[1.0, 2.0, 3.0], [-0.5, 0.3, 1.0]])
        assert forces_cosine_similarity(f, -f) == pytest.approx(-1.0)

    def test_orthogonal(self):
        f1 = np.array([[1.0, 0.0, 0.0]])
        f2 = np.array([[0.0, 1.0, 0.0]])
        assert forces_cosine_similarity(f1, f2) == pytest.approx(0.0, abs=1e-10)

    def test_both_zero(self):
        z = np.zeros((2, 3))
        assert forces_cosine_similarity(z, z) == pytest.approx(1.0)

    def test_one_zero(self):
        f = np.array([[1.0, 0.0, 0.0]])
        z = np.zeros_like(f)
        assert forces_cosine_similarity(f, z) == pytest.approx(0.0)


# ------------------------------------------------------------------ #
#  confirm_duplicates with force gate
# ------------------------------------------------------------------ #


class TestConfirmDuplicatesForces:
    def test_forces_agree(self):
        cu = bulk("Cu", "fcc", a=3.615)
        forces = np.array([[0.1, 0.2, 0.3]])
        nearby = [_nearby(cu, -3.0, forces=forces)]
        result = confirm_duplicates(
            cu,
            nearby,
            use_forces=True,
            candidate_forces=forces,
            force_cosine_threshold=0.95,
        )
        assert result == pytest.approx(-3.0)

    def test_forces_disagree(self):
        cu = bulk("Cu", "fcc", a=3.615)
        f_cand = np.array([[1.0, 0.0, 0.0]])
        f_near = np.array([[0.0, 1.0, 0.0]])
        nearby = [_nearby(cu, -3.0, forces=f_near)]
        result = confirm_duplicates(
            cu,
            nearby,
            use_forces=True,
            candidate_forces=f_cand,
            force_cosine_threshold=0.95,
        )
        assert result is None

    def test_forces_disabled_ignores_forces(self):
        cu = bulk("Cu", "fcc", a=3.615)
        f_cand = np.array([[1.0, 0.0, 0.0]])
        f_near = np.array([[0.0, 1.0, 0.0]])
        nearby = [_nearby(cu, -3.0, forces=f_near)]
        result = confirm_duplicates(
            cu,
            nearby,
            use_forces=False,
            candidate_forces=f_cand,
        )
        assert result == pytest.approx(-3.0)

    def test_missing_nearby_forces_conservative(self):
        """If a nearby entry has no forces, it's not confirmed (conservative)."""
        cu = bulk("Cu", "fcc", a=3.615)
        f_cand = np.array([[1.0, 0.0, 0.0]])
        nearby = [_nearby(cu, -3.0, forces=None)]
        result = confirm_duplicates(
            cu,
            nearby,
            use_forces=True,
            candidate_forces=f_cand,
            force_cosine_threshold=0.95,
        )
        assert result is None

    def test_missing_candidate_forces_returns_none(self):
        cu = bulk("Cu", "fcc", a=3.615)
        forces = np.array([[0.1, 0.2, 0.3]])
        nearby = [_nearby(cu, -3.0, forces=forces)]
        result = confirm_duplicates(
            cu,
            nearby,
            use_forces=True,
            candidate_forces=None,
            force_cosine_threshold=0.95,
        )
        assert result is None


# ------------------------------------------------------------------ #
#  Multi-stage: pymatgen + forces
# ------------------------------------------------------------------ #


class TestMultiStage:
    def test_all_three_stages_pass(self):
        """Vector close + pymatgen match + forces agree -> confirmed."""
        cu = bulk("Cu", "fcc", a=3.615)
        cu_dup = bulk("Cu", "fcc", a=3.615)
        forces = np.array([[0.1, 0.2, 0.3]])
        nearby = [_nearby(cu_dup, -3.0, forces=forces)]
        result = confirm_duplicates(
            cu,
            nearby,
            use_pymatgen=True,
            use_forces=True,
            candidate_forces=forces,
            force_cosine_threshold=0.95,
        )
        assert result == pytest.approx(-3.0)

    def test_pymatgen_pass_forces_fail(self):
        """Pymatgen says same but forces disagree -> not confirmed."""
        cu = bulk("Cu", "fcc", a=3.615)
        cu_dup = bulk("Cu", "fcc", a=3.615)
        f_cand = np.array([[1.0, 0.0, 0.0]])
        f_near = np.array([[-1.0, 0.0, 0.0]])
        nearby = [_nearby(cu_dup, -3.0, forces=f_near)]
        result = confirm_duplicates(
            cu,
            nearby,
            use_pymatgen=True,
            use_forces=True,
            candidate_forces=f_cand,
            force_cosine_threshold=0.95,
        )
        assert result is None

    def test_pymatgen_fail_forces_pass(self):
        """Pymatgen rejects (BCC vs FCC) even though forces agree."""
        cu_fcc = bulk("Cu", "fcc", a=3.615)
        cu_bcc = bulk("Cu", "bcc", a=2.87)
        forces = np.array([[0.1, 0.2, 0.3]])
        nearby = [_nearby(cu_bcc, -3.0, forces=forces)]
        result = confirm_duplicates(
            cu_fcc,
            nearby,
            use_pymatgen=True,
            use_forces=True,
            candidate_forces=forces,
            force_cosine_threshold=0.95,
        )
        assert result is None
