"""Integration test for the main relaxation + filtering processing loop.

Uses real SurrealDB, real SOAP descriptors, and real EMT physics.
Only the calculator factory is mocked (to avoid loading MatterSim/NequIP).
"""

from unittest.mock import patch

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from rapmat.core.csp import run_processing_loop
from rapmat.storage import SOAPDescriptor, SurrealDBStore


@pytest.fixture
def loop_env(tmp_path):
    """Store with 3 pre-generated Cu candidates ready for the processing loop."""
    descriptor = SOAPDescriptor(species=["Cu"], n_max=4, l_max=3)
    store = SurrealDBStore.from_path(tmp_path / "loop_db")
    store.register_descriptor(descriptor.descriptor_id(), descriptor.dimension())

    run_name = "loop-run"
    config = {
        "formula": {"Cu": 1},
        "calculator": "MATTERSIM",
        "calculator_config": {},
        "domain": "bulk",
        "skip_not_converged": False,
        "thickness_cutoff": None,
        "dedup_threshold": 1e-2,
        "symprec": 1e-3,
    }
    store.create_study(study_id=f"study-{run_name}", system="Cu", domain="bulk", calculator="MATTERSIM", config=config)
    store.create_run(name=run_name, study_id=f"study-{run_name}")

    zero_vec = np.zeros(descriptor.dimension(), dtype=np.float32)

    # Cu FCC at different lattice constants + one BCC
    for idx, (struct_type, a) in enumerate(
        [("fcc", 3.7), ("fcc", 3.8), ("bcc", 2.87)], start=1
    ):
        atoms = bulk("Cu", struct_type, a=a)
        store.add_candidate(atoms, zero_vec, run_name, f"{run_name}/{idx}")

    return {
        "store": store,
        "run_name": run_name,
        "config": config,
        "workdir": tmp_path,
        "descriptor": descriptor,
    }


@patch("rapmat.calculators.factory.load_calculator")
def test_processing_loop_end_to_end(mock_load_calc, loop_env):
    """Full pipeline: relax -> filter -> dedup -> store, using EMT."""
    mock_load_calc.return_value = EMT()

    run_processing_loop(
        run_name=loop_env["run_name"],
        store=loop_env["store"],
        config=loop_env["config"],
        workdir_path=loop_env["workdir"],
        descriptor=loop_env["descriptor"],
    )

    store = loop_env["store"]
    run_name = loop_env["run_name"]

    # No candidates left unprocessed
    assert len(store.get_unrelaxed_candidates(run_name)) == 0

    # All 3 candidates should be accounted for
    counts = store.count_by_status(run_name)
    total = sum(counts.values())
    assert total == 3

    # No errors expected with EMT on simple Cu structures
    assert counts.get("error", 0) == 0

    # At least one structure should be successfully relaxed
    relaxed = store.get_run_structures(run_name, status="relaxed")
    assert len(relaxed) >= 1

    # Verify relaxed structures have reasonable metadata
    for r in relaxed:
        # EMT energies can be positive (relative to isolated atoms)
        assert abs(r["energy_per_atom"]) < 100.0  # Reasonable magnitude
        assert r["fmax"] >= 0
        assert r["converged"] is True
        assert r["final_atoms"] is not None
        assert r["final_spg"] != ""


@patch("rapmat.calculators.factory.load_calculator")
def test_candidate_dedup_skips_relaxation(mock_load_calc, tmp_path):
    """Candidate dedup should discard duplicates before relaxation."""
    descriptor = SOAPDescriptor(species=["Cu"], n_max=4, l_max=3)
    store = SurrealDBStore.from_path(tmp_path / "cand_dedup_db")
    store.register_descriptor(descriptor.descriptor_id(), descriptor.dimension())

    run_name = "cand-dedup-run"
    config = {
        "formula": {"Cu": 1},
        "calculator": "MATTERSIM",
        "calculator_config": {},
        "domain": "bulk",
        "skip_not_converged": False,
        "thickness_cutoff": None,
        "dedup": True,
        "dedup_threshold": 5.0,
        "symprec": 1e-3,
    }
    store.create_study(study_id=f"study-{run_name}", system="Cu", domain="bulk", calculator="MATTERSIM", config=config)
    store.create_run(name=run_name, study_id=f"study-{run_name}")

    cu = bulk("Cu", "fcc", a=3.615)
    vec = descriptor.compute(cu)

    # First candidate: generated with vector so it's findable
    store.add_candidate(cu, vec, run_name, f"{run_name}/1")

    # Second candidate: near-identical, should be caught by candidate dedup
    cu2 = bulk("Cu", "fcc", a=3.616)
    vec2 = descriptor.compute(cu2)
    store.add_candidate(cu2, vec2, run_name, f"{run_name}/2")

    mock_load_calc.return_value = EMT()

    run_processing_loop(
        run_name=run_name,
        store=store,
        config=config,
        workdir_path=tmp_path,
        descriptor=descriptor,
    )

    counts = store.count_by_status(run_name)
    # With a wide dedup threshold, at most 1 should be relaxed;
    # the other(s) should be discarded as candidate duplicates.
    assert counts.get("relaxed", 0) >= 1
    assert counts.get("discarded", 0) >= 1
    assert counts.get("error", 0) == 0


@patch("rapmat.calculators.factory.load_calculator")
def test_dedup_flag_disabled_keeps_duplicates(mock_load_calc, tmp_path):
    """When dedup=False, candidate/relaxed dedup blocks are bypassed."""
    descriptor = SOAPDescriptor(species=["Cu"], n_max=4, l_max=3)
    store = SurrealDBStore.from_path(tmp_path / "no_dedup_db")
    store.register_descriptor(descriptor.descriptor_id(), descriptor.dimension())

    run_name = "no-dedup-run"
    config = {
        "formula": {"Cu": 1},
        "calculator": "MATTERSIM",
        "calculator_config": {},
        "domain": "bulk",
        "skip_not_converged": False,
        "thickness_cutoff": None,
        "dedup": False,
        "dedup_threshold": 5.0,
        "symprec": 1e-3,
    }
    store.create_study(study_id=f"study-{run_name}", system="Cu", domain="bulk", calculator="MATTERSIM", config=config)
    store.create_run(name=run_name, study_id=f"study-{run_name}")

    cu = bulk("Cu", "fcc", a=3.615)
    vec = descriptor.compute(cu)

    # First candidate
    store.add_candidate(cu, vec, run_name, f"{run_name}/1")

    # Second near-duplicate candidate that would normally be deduplicated
    cu2 = bulk("Cu", "fcc", a=3.616)
    vec2 = descriptor.compute(cu2)
    store.add_candidate(cu2, vec2, run_name, f"{run_name}/2")

    mock_load_calc.return_value = EMT()

    run_processing_loop(
        run_name=run_name,
        store=store,
        config=config,
        workdir_path=tmp_path,
        descriptor=descriptor,
    )

    counts = store.count_by_status(run_name)
    # With dedup disabled, both candidates should be processed and not discarded as duplicates.
    assert counts.get("relaxed", 0) >= 2
    assert counts.get("discarded", 0) == 0
    assert counts.get("error", 0) == 0


@patch("rapmat.calculators.factory.load_calculator")
def test_processing_loop_convergence_filter(mock_load_calc, tmp_path):
    """With skip_not_converged=True, unconverged structures are discarded."""
    descriptor = SOAPDescriptor(species=["Cu"], n_max=4, l_max=3)
    store = SurrealDBStore.from_path(tmp_path / "conv_db")
    store.register_descriptor(descriptor.descriptor_id(), descriptor.dimension())

    run_name = "conv-run"
    config = {
        "formula": {"Cu": 1},
        "calculator": "MATTERSIM",
        "calculator_config": {},
        "domain": "bulk",
        "skip_not_converged": True,
        "thickness_cutoff": None,
        "dedup_threshold": 1e-2,
        "symprec": 1e-3,
    }
    store.create_study(study_id=f"study-{run_name}", system="Cu", domain="bulk", calculator="MATTERSIM", config=config)
    store.create_run(name=run_name, study_id=f"study-{run_name}")

    zero_vec = np.zeros(descriptor.dimension(), dtype=np.float32)

    # Cu FCC that should converge easily
    atoms = bulk("Cu", "fcc", a=3.7)
    store.add_candidate(atoms, zero_vec, run_name, f"{run_name}/1")

    mock_load_calc.return_value = EMT()

    run_processing_loop(
        run_name=run_name,
        store=store,
        config=config,
        workdir_path=tmp_path,
        descriptor=descriptor,
    )

    counts = store.count_by_status(run_name)
    # The well-behaved Cu FCC should converge and be kept
    assert counts.get("relaxed", 0) >= 1
    assert counts.get("error", 0) == 0
