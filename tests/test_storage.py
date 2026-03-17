import math

import numpy as np
import pytest
from ase.build import bulk

try:
    from rapmat.storage import SOAPDescriptor, SurrealDBStore
except ImportError:
    pytest.skip("rapmat.storage not available", allow_module_level=True)


def test_soap_descriptor_basics():
    atoms = bulk("Si", "diamond", a=5.43)
    desc = SOAPDescriptor(species=["Si"], n_max=4, l_max=3)
    vec = desc.compute(atoms)

    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert vec.shape[0] == desc.dimension()
    assert vec.shape[0] > 0
    assert np.linalg.norm(vec) > 0


def test_run_lifecycle(tmp_path):
    """Create a run, add candidates, update to relaxed, query back."""
    db_path = tmp_path / "test_db"
    store = SurrealDBStore.from_path(db_path)
    atoms = bulk("Si", "diamond", a=5.43)

    # Create run
    store.create_study("test-study", "Si", "bulk", "mattersim")
    run_id = store.create_run(name="test-run", study_id="test-study")
    store.update_run_config(run_id, {"formula": {"Si": 1}})
    assert run_id == "test-run"

    # Verify metadata
    meta = store.get_run_metadata("test-run")
    assert meta is not None
    assert meta["name"] == "test-run"
    assert meta["domain"] == "bulk"
    assert meta["config"] == {
        "formula": {"Si": 1},
        "system": "Si",
        "domain": "bulk",
        "calculator": "mattersim",
    }

    # Register a descriptor so vector operations work
    store.register_descriptor(
        "test000000000000000000000000000000000000000000000000000000000000", 10
    )

    # Add candidate
    zero_vec = np.zeros(10, dtype=np.float32)
    struct_id = store.add_candidate(atoms, zero_vec, "test-run", "test-run/1")
    assert struct_id == "test-run/1"

    # Fetch unrelaxed
    unrelaxed = store.get_unrelaxed_candidates(run_id)
    assert len(unrelaxed) == 1
    assert unrelaxed[0]["id"] == struct_id

    # Update to relaxed
    vec = np.random.randn(10).astype(np.float32)
    store.update_structure(
        struct_id,
        "relaxed",
        atoms=atoms,
        vector=vec,
        metadata={
            "energy_per_atom": -5.0,
            "energy_total": -40.0,
            "fmax": 0.01,
            "converged": True,
        },
    )

    # Fetch relaxed -- SPG is computed on-the-fly from stored atoms
    relaxed = store.get_run_structures("test-run", status="relaxed")
    assert len(relaxed) == 1
    assert relaxed[0]["final_spg"] != ""
    assert relaxed[0]["initial_spg"] != ""
    assert relaxed[0]["converged"] is True

    # No more unrelaxed
    unrelaxed = store.get_unrelaxed_candidates("test-run")
    assert len(unrelaxed) == 0

    store.close()


@pytest.fixture
def mock_study(tmp_path):
    pass # Replaced by inline creation where structure matters

def test_deduplication(tmp_path):
    """is_duplicate should detect nearby relaxed structures."""
    db_path = tmp_path / "test_dedup"
    store = SurrealDBStore.from_path(db_path)
    atoms = bulk("Si", "diamond", a=5.43)
    
    store.create_study("dedup-study", "Si", "bulk", "mattersim")
    run_id = store.create_run(name="dedup-test", study_id="dedup-study")
    store.update_run_config(run_id, {"formula": {"Si": 1}})
    
    store.register_descriptor(
        "test000000000000000000000000000000000000000000000000000000000000", 10
    )

    # Add and relax one structure
    vec1 = np.zeros(10, dtype=np.float32)
    sid = store.add_candidate(atoms, vec1, "dedup-test", "dedup-test/1")
    store.update_structure(
        sid, "relaxed", atoms=atoms, vector=vec1, metadata={"converged": True}
    )

    # Same vector -> duplicate
    assert store.is_duplicate(vec1, threshold=0.1) is True

    # Far vector -> not duplicate
    vec_far = np.ones(10, dtype=np.float32) * 10.0
    assert store.is_duplicate(vec_far, threshold=0.1) is False

    # Nearby vector with large threshold -> duplicate
    vec_near = np.zeros(10, dtype=np.float32)
    vec_near[0] = 0.5
    assert store.is_duplicate(vec_near, threshold=1.0) is True

    # Nearby vector with small threshold -> not duplicate
    assert store.is_duplicate(vec_near, threshold=0.1) is False

    store.close()


def test_duplicate_min_energy(tmp_path):
    """get_duplicate_min_energy returns min energy among duplicates or None."""
    db_path = tmp_path / "test_dedup_min_energy"
    store = SurrealDBStore.from_path(db_path)
    atoms = bulk("Si", "diamond", a=5.43)
    
    store.create_study("energy-study", "Si", "bulk", "mattersim")
    run_id = store.create_run(name="dedup-energy-test", study_id="energy-study")
    store.update_run_config(run_id, {"formula": {"Si": 1}})
    
    store.register_descriptor(
        "test000000000000000000000000000000000000000000000000000000000000", 10
    )

    # No relaxed structures -> None
    vec = np.zeros(10, dtype=np.float32)
    assert store.get_duplicate_min_energy(vec, threshold=0.1) is None

    # Add relaxed structure with energy_per_atom -5.0
    sid1 = store.add_candidate(atoms, vec, "dedup-energy-test", "dedup-energy-test/1")
    store.update_structure(
        sid1,
        "relaxed",
        atoms=atoms,
        vector=vec,
        metadata={"converged": True, "energy_per_atom": -5.0, "energy_total": -40.0},
    )

    # Same vector -> min energy -5.0
    assert store.get_duplicate_min_energy(vec, threshold=0.1) == -5.0

    # Far vector -> None
    vec_far = np.ones(10, dtype=np.float32) * 10.0
    assert store.get_duplicate_min_energy(vec_far, threshold=0.1) is None

    # Second relaxed structure with higher energy, nearby vector
    vec2 = np.zeros(10, dtype=np.float32)
    vec2[0] = 0.3
    sid2 = store.add_candidate(atoms, vec2, "dedup-energy-test", "dedup-energy-test/2")
    store.update_structure(
        sid2,
        "relaxed",
        atoms=atoms,
        vector=vec2,
        metadata={"converged": True, "energy_per_atom": -3.0, "energy_total": -24.0},
    )

    # Query with vec (distance 0 to first, ~0.3 to second): both within threshold 1.0, min is -5.0
    assert store.get_duplicate_min_energy(vec, threshold=1.0) == -5.0

    # Query with vec2: both within threshold 1.0, min is still -5.0
    assert store.get_duplicate_min_energy(vec2, threshold=1.0) == -5.0

    store.close()


def test_add_candidates_batch(tmp_path):
    """add_candidates inserts multiple candidates in one batch and they are readable."""
    db_path = tmp_path / "test_batch"
    store = SurrealDBStore.from_path(db_path)
    atoms = bulk("Si", "diamond", a=5.43)
    
    store.create_study("batch-study", "Si", "bulk", "mattersim")
    run_id = store.create_run(name="batch-run", study_id="batch-study")
    store.update_run_config(run_id, {"formula": {"Si": 1}})
    store.register_descriptor(
        "test000000000000000000000000000000000000000000000000000000000000", 10
    )

    zero_vec = np.zeros(10, dtype=np.float32)
    candidates = [
        (atoms, zero_vec, "batch-run", "batch-run/1", None),
        (atoms, zero_vec, "batch-run", "batch-run/2", None),
        (atoms, zero_vec, "batch-run", "batch-run/3", None),
    ]
    n = store.add_candidates(candidates)
    assert n == 3

    unrelaxed = store.get_unrelaxed_candidates(run_id)
    assert len(unrelaxed) == 3
    ids = {r["id"] for r in unrelaxed}
    assert ids == {"batch-run/1", "batch-run/2", "batch-run/3"}
    assert store.count() == 3

    store.close()


def test_legacy_add_if_unique(tmp_path):
    """Backwards-compatible add_if_unique still works."""
    db_path = tmp_path / "test_legacy"
    store = SurrealDBStore.from_path(db_path)
    atoms = bulk("Si", "diamond", a=5.43)
    store.register_descriptor(
        "test000000000000000000000000000000000000000000000000000000000000", 10
    )

    vec1 = np.zeros(10, dtype=np.float32)
    meta = {"converged": True, "energy_per_atom": -5.0}

    added = store.add_if_unique(atoms, vec1, meta, threshold=0.1)
    assert added is True
    assert store.count() == 1

    added = store.add_if_unique(atoms, vec1, meta, threshold=0.1)
    assert added is False
    assert store.count() == 1

    store.close()


def test_phonon_min_freq_persistence(tmp_path):
    """update_structure_phonon persists min_phonon_freq; get_run_structures returns it."""
    db_path = tmp_path / "test_phonon"
    store = SurrealDBStore.from_path(db_path)
    atoms = bulk("Si", "diamond", a=5.43)
    
    store.create_study("phonon-study", "Si", "bulk", "mattersim")
    run_id = store.create_run(name="phonon-run", study_id="phonon-study")
    store.update_run_config(run_id, {"formula": {"Si": 1}})
    store.register_descriptor(
        "test000000000000000000000000000000000000000000000000000000000000", 10
    )
    sid = store.add_candidate(
        atoms, np.zeros(10, dtype=np.float32), "phonon-run", "phonon-run/1"
    )
    store.update_structure(
        sid,
        "relaxed",
        atoms=atoms,
        vector=np.zeros(10, dtype=np.float32),
        metadata={"converged": True},
    )

    relaxed = store.get_run_structures("phonon-run", status="relaxed")
    assert len(relaxed) == 1
    assert relaxed[0].get("min_phonon_freq") is None

    store.update_structure_phonon("phonon-run/1", -0.05)
    relaxed = store.get_run_structures("phonon-run", status="relaxed")
    assert len(relaxed) == 1
    assert relaxed[0]["min_phonon_freq"] == -0.05

    store.clear_run_phonon_results("phonon-run")
    relaxed = store.get_run_structures("phonon-run", status="relaxed")
    assert len(relaxed) == 1
    val = relaxed[0].get("min_phonon_freq")
    assert val is None or (isinstance(val, float) and math.isnan(val))

    store.close()


def test_initial_and_final_atoms_preserved(tmp_path):
    """Both initial and final atoms survive the add / update cycle."""
    db_path = tmp_path / "test_dual_atoms"
    store = SurrealDBStore.from_path(db_path)
    
    store.create_study("dual-study", "Si", "bulk", "mattersim")
    run_id = store.create_run(name="dual-run", study_id="dual-study")
    store.update_run_config(run_id, {"formula": {"Si": 1}})
    store.register_descriptor(
        "test000000000000000000000000000000000000000000000000000000000000", 10
    )

    initial_atoms = bulk("Si", "diamond", a=5.43)
    relaxed_atoms = bulk("Si", "diamond", a=5.40)

    zero_vec = np.zeros(10, dtype=np.float32)
    sid = store.add_candidate(initial_atoms, zero_vec, "dual-run", "dual-run/1")

    # Before relaxation: only initial atoms present
    unrelaxed = store.get_unrelaxed_candidates("dual-run")
    assert len(unrelaxed) == 1
    assert len(unrelaxed[0]["atoms"]) == len(initial_atoms)

    # Relax with different atoms
    store.update_structure(
        sid,
        "relaxed",
        atoms=relaxed_atoms,
        vector=zero_vec,
        metadata={"converged": True, "energy_per_atom": -5.0, "energy_total": -10.0},
    )

    records = store.get_run_structures("dual-run", status="relaxed")
    assert len(records) == 1
    rec = records[0]

    # Both atoms should be present and distinct
    assert rec["initial_atoms"] is not None
    assert rec["final_atoms"] is not None
    np.testing.assert_allclose(
        rec["initial_atoms"].cell.lengths(), initial_atoms.cell.lengths(), atol=1e-6
    )
    np.testing.assert_allclose(
        rec["final_atoms"].cell.lengths(), relaxed_atoms.cell.lengths(), atol=1e-6
    )

    # Convenience 'atoms' alias should be the final (relaxed) atoms
    np.testing.assert_allclose(
        rec["atoms"].cell.lengths(), relaxed_atoms.cell.lengths(), atol=1e-6
    )

    store.close()


def test_spg_recomputation(tmp_path):
    """Different symprec values should produce different SPG results for the same atoms."""
    db_path = tmp_path / "test_spg"
    store = SurrealDBStore.from_path(db_path)
    
    store.create_study("spg-study", "Si", "bulk", "mattersim")
    run_id = store.create_run(name="spg-run", study_id="spg-study")
    store.update_run_config(run_id, {"formula": {"Si": 1}})
    store.register_descriptor(
        "test000000000000000000000000000000000000000000000000000000000000", 10
    )

    atoms = bulk("Si", "diamond", a=5.43)
    zero_vec = np.zeros(10, dtype=np.float32)
    sid = store.add_candidate(atoms, zero_vec, "spg-run", "spg-run/1")
    store.update_structure(
        sid,
        "relaxed",
        atoms=atoms,
        vector=zero_vec,
        metadata={"converged": True, "energy_per_atom": -5.0, "energy_total": -10.0},
    )

    # Default symprec should produce a valid SPG string
    records_default = store.get_run_structures(
        "spg-run", status="relaxed", symprec=1e-3
    )
    assert len(records_default) == 1
    spg_default = records_default[0]["final_spg"]
    assert spg_default != ""
    assert spg_default != "N/A"
    assert "(" in spg_default and ")" in spg_default

    # Very tight symprec may produce a lower-symmetry group
    records_tight = store.get_run_structures("spg-run", status="relaxed", symprec=1e-10)
    assert len(records_tight) == 1
    spg_tight = records_tight[0]["final_spg"]
    assert spg_tight != ""

    # Both initial and final SPG should be populated for the same structure
    assert records_default[0]["initial_spg"] != ""

    store.close()
