"""Tests for the post-hoc dedup analysis / simulation module."""

import numpy as np
import pytest
from ase.build import bulk

from rapmat.core.dedup_analysis import (
    DedupSimulationResult,
    compute_pairwise_distances,
    find_threshold_for_survival,
    plot_distance_histogram,
    simulate_deduplication,
)
from rapmat.storage import SOAPDescriptor, SurrealDBStore


def _make_entry(struct_id, atoms, energy, vector, forces=None):
    return {
        "id": struct_id,
        "atoms": atoms,
        "energy_per_atom": energy,
        "vector": vector,
        "forces": forces,
    }


# ------------------------------------------------------------------ #
#  compute_pairwise_distances
# ------------------------------------------------------------------ #


class TestPairwiseDistances:
    def test_identical_vectors(self):
        vecs = np.array([[1.0, 0.0], [1.0, 0.0]])
        d = compute_pairwise_distances(vecs)
        assert len(d) == 1
        assert d[0] == pytest.approx(0.0)

    def test_known_distance(self):
        vecs = np.array([[0.0, 0.0], [3.0, 4.0]])
        d = compute_pairwise_distances(vecs)
        assert d[0] == pytest.approx(5.0)

    def test_three_vectors(self):
        vecs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        d = compute_pairwise_distances(vecs)
        assert len(d) == 3


# ------------------------------------------------------------------ #
#  simulate_deduplication
# ------------------------------------------------------------------ #


class TestSimulateDedup:
    @pytest.fixture
    def soap(self):
        return SOAPDescriptor(species=["Cu"], n_max=4, l_max=3)

    def test_empty_input(self):
        result = simulate_deduplication([], threshold=0.1)
        assert result.total == 0
        assert result.kept == 0

    def test_no_vectors(self):
        cu = bulk("Cu", "fcc", a=3.615)
        entries = [_make_entry("s/1", cu, -3.0, None)]
        result = simulate_deduplication(entries, threshold=0.1)
        assert result.kept == 1
        assert result.final_dropped == 0

    def test_identical_structures_dropped(self, soap):
        cu = bulk("Cu", "fcc", a=3.615)
        vec = soap.compute(cu)
        entries = [
            _make_entry("s/1", cu, -3.0, vec.copy()),
            _make_entry("s/2", cu, -2.0, vec.copy()),
        ]
        result = simulate_deduplication(entries, threshold=1.0)
        assert result.kept == 1
        assert result.final_dropped == 1
        assert "s/1" in result.kept_ids

    def test_distinct_structures_kept(self, soap):
        cu_fcc = bulk("Cu", "fcc", a=3.615)
        cu_bcc = bulk("Cu", "bcc", a=2.87)
        vec_fcc = soap.compute(cu_fcc)
        vec_bcc = soap.compute(cu_bcc)
        entries = [
            _make_entry("s/1", cu_fcc, -3.0, vec_fcc),
            _make_entry("s/2", cu_bcc, -2.5, vec_bcc),
        ]
        result = simulate_deduplication(entries, threshold=1e-5)
        assert result.kept == 2
        assert result.final_dropped == 0

    def test_pymatgen_rescues_collision(self, soap):
        cu_fcc = bulk("Cu", "fcc", a=3.615)
        cu_bcc = bulk("Cu", "bcc", a=2.87)
        vec = soap.compute(cu_fcc)
        entries = [
            _make_entry("s/1", cu_fcc, -3.0, vec.copy()),
            _make_entry("s/2", cu_bcc, -2.5, vec.copy()),
        ]
        result = simulate_deduplication(
            entries,
            threshold=999.0,
            use_pymatgen=True,
        )
        assert result.kept == 2
        assert result.rescued_by_pymatgen >= 1

    def test_forces_rescue_different_gradients(self, soap):
        cu = bulk("Cu", "fcc", a=3.615)
        vec = soap.compute(cu)
        f1 = np.array([[1.0, 0.0, 0.0]])
        f2 = np.array([[0.0, 1.0, 0.0]])
        entries = [
            _make_entry("s/1", cu, -3.0, vec.copy(), forces=f1),
            _make_entry("s/2", cu, -2.5, vec.copy(), forces=f2),
        ]
        result = simulate_deduplication(
            entries,
            threshold=999.0,
            use_forces=True,
            force_cosine_threshold=0.95,
        )
        assert result.kept == 2
        assert result.rescued_by_forces >= 1

    def test_progress_callback(self, soap):
        cu = bulk("Cu", "fcc", a=3.615)
        vec = soap.compute(cu)
        entries = [_make_entry("s/1", cu, -3.0, vec)]
        calls = []
        simulate_deduplication(
            entries,
            threshold=0.1,
            progress_callback=lambda c, t, l: calls.append((c, t)),
        )
        assert len(calls) > 0

    def test_lower_energy_kept(self, soap):
        """The structure with lower energy is kept when two are identical."""
        cu = bulk("Cu", "fcc", a=3.615)
        vec = soap.compute(cu)
        entries = [
            _make_entry("s/high", cu, -1.0, vec.copy()),
            _make_entry("s/low", cu, -5.0, vec.copy()),
        ]
        result = simulate_deduplication(entries, threshold=1.0)
        assert "s/low" in result.kept_ids
        assert "s/high" in result.dropped_ids


# ------------------------------------------------------------------ #
#  find_threshold_for_survival
# ------------------------------------------------------------------ #


class TestFindThresholdForSurvival:
    @pytest.fixture
    def soap(self):
        return SOAPDescriptor(species=["Cu"], n_max=4, l_max=3)

    def test_full_survival(self, soap):
        cu = bulk("Cu", "fcc", a=3.615)
        vec = soap.compute(cu)
        entries = [_make_entry("s/1", cu, -3.0, vec)]
        thresh, kept = find_threshold_for_survival(entries, 1.0, 10.0)
        assert kept == 1
        assert thresh == 0.0

    def test_no_vectors(self):
        cu = bulk("Cu", "fcc", a=3.615)
        entries = [_make_entry("s/1", cu, -3.0, None)]
        thresh, kept = find_threshold_for_survival(entries, 0.5, 10.0)
        assert kept == 1

    def test_identical_structures_low_survival(self, soap):
        cu = bulk("Cu", "fcc", a=3.615)
        vec = soap.compute(cu)
        entries = [
            _make_entry(f"s/{i}", cu, -3.0 + i * 0.1, vec.copy()) for i in range(10)
        ]
        thresh, kept = find_threshold_for_survival(entries, 0.1, 10.0)
        assert kept <= 3

    def test_distinct_structures_high_survival(self, soap):
        cu_fcc = bulk("Cu", "fcc", a=3.615)
        cu_bcc = bulk("Cu", "bcc", a=2.87)
        vec_fcc = soap.compute(cu_fcc)
        vec_bcc = soap.compute(cu_bcc)
        entries = [
            _make_entry("s/fcc", cu_fcc, -3.0, vec_fcc),
            _make_entry("s/bcc", cu_bcc, -2.5, vec_bcc),
        ]
        thresh, kept = find_threshold_for_survival(entries, 0.95, 1000.0)
        assert kept == 2

    def test_monotonic_survival(self, soap):
        """Lower survival target requires a larger (or equal) threshold."""
        entries = []
        for i in range(30):
            cu = bulk("Cu", "fcc", a=3.615 + i * 0.05)
            vec = soap.compute(cu)
            entries.append(_make_entry(f"s/{i}", cu, -3.0 + i * 0.1, vec))
        prev_thresh = -1.0
        for target in [0.9, 0.5, 0.1]:
            thresh, _ = find_threshold_for_survival(entries, target, 500.0)
            assert thresh >= prev_thresh - 1e-6
            prev_thresh = thresh


# ------------------------------------------------------------------ #
#  plot_distance_histogram
# ------------------------------------------------------------------ #


class TestPlotHistogram:
    def test_saves_png(self, tmp_path):
        distances = np.array([0.1, 0.5, 1.0, 2.0, 3.0, 0.8])
        out = tmp_path / "hist.png"
        plot_distance_histogram(distances, threshold=0.5, save_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_saves_svg(self, tmp_path):
        distances = np.array([0.1, 0.5, 1.0])
        out = tmp_path / "hist.svg"
        plot_distance_histogram(distances, save_path=out)
        assert out.exists()


# ------------------------------------------------------------------ #
#  Store integration: get_structures_for_analysis
# ------------------------------------------------------------------ #


class TestStoreAnalysis:
    @pytest.fixture
    def soap(self):
        return SOAPDescriptor(species=["Cu"], n_max=4, l_max=3)

    def test_returns_relaxed_structures(self, tmp_path, soap):
        store = SurrealDBStore.from_path(tmp_path / "analysis_db")
        store.register_descriptor(soap.descriptor_id(), soap.dimension())
        store.create_study(
            study_id="run1",
            system="Test",
            domain="bulk",
            calculator="MATTERSIM",
            config={},
        )
        store.create_run(name="run1", study_id="run1")

        cu = bulk("Cu", "fcc", a=3.615)
        vec = soap.compute(cu)
        store.add_candidate(cu, vec, "run1", "run1/1")
        store.update_structure(
            "run1/1",
            "relaxed",
            atoms=cu,
            vector=vec,
            metadata={"energy_per_atom": -3.0, "energy_total": -3.0, "converged": True},
        )

        results = store.get_structures_for_analysis("run1", statuses=("relaxed",))
        assert len(results) == 1
        assert results[0]["energy_per_atom"] == pytest.approx(-3.0)
        assert "vector" not in results[0]
        assert results[0]["atoms"] is not None

    def test_returns_generated_structures(self, tmp_path, soap):
        store = SurrealDBStore.from_path(tmp_path / "analysis_gen_db")
        store.register_descriptor(soap.descriptor_id(), soap.dimension())
        store.create_study(
            study_id="run1",
            system="Test",
            domain="bulk",
            calculator="MATTERSIM",
            config={},
        )
        store.create_run(name="run1", study_id="run1")

        cu = bulk("Cu", "fcc", a=3.615)
        vec = soap.compute(cu)
        store.add_candidate(cu, vec, "run1", "run1/1")

        results = store.get_structures_for_analysis("run1", statuses=("generated",))
        assert len(results) == 1
        assert results[0]["id"] == "run1/1"

    def test_filters_by_status(self, tmp_path, soap):
        store = SurrealDBStore.from_path(tmp_path / "analysis_filter_db")
        store.register_descriptor(soap.descriptor_id(), soap.dimension())
        store.create_study(
            study_id="run1",
            system="Test",
            domain="bulk",
            calculator="MATTERSIM",
            config={},
        )
        store.create_run(name="run1", study_id="run1")

        cu = bulk("Cu", "fcc", a=3.615)
        vec = soap.compute(cu)
        store.add_candidate(cu, vec, "run1", "run1/1")
        store.update_structure(
            "run1/1",
            "relaxed",
            atoms=cu,
            vector=vec,
            metadata={"energy_per_atom": -3.0, "energy_total": -3.0, "converged": True},
        )

        cu2 = bulk("Cu", "fcc", a=3.62)
        vec2 = soap.compute(cu2)
        store.add_candidate(cu2, vec2, "run1", "run1/2")

        relaxed = store.get_structures_for_analysis("run1", statuses=("relaxed",))
        assert len(relaxed) == 1

        generated = store.get_structures_for_analysis("run1", statuses=("generated",))
        assert len(generated) == 1

        both = store.get_structures_for_analysis(
            "run1", statuses=("generated", "relaxed")
        )
        assert len(both) == 2
