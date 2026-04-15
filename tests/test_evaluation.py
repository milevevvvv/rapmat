"""Unit tests for rapmat.core.evaluation metric functions."""

import pytest

from rapmat.core.evaluation import (compute_ranking_metrics,
                                    compute_stability_metrics)

# ------------------------------------------------------------------ #
#  compute_ranking_metrics
# ------------------------------------------------------------------ #


class TestRankingMetrics:
    def test_perfect_agreement(self):
        results = [
            {"mlip_epa": -5.0, "ref_epa": -5.0},
            {"mlip_epa": -4.0, "ref_epa": -4.0},
            {"mlip_epa": -3.0, "ref_epa": -3.0},
        ]
        m = compute_ranking_metrics(results, stable_only=False)
        assert m["kendall_tau"] == pytest.approx(1.0)
        assert m["mae_epa"] == pytest.approx(0.0)
        assert m["n_structures"] == 3
        assert m["stable_only_applied"] is False

    def test_reversed_ranking(self):
        results = [
            {"mlip_epa": -5.0, "ref_epa": -3.0},
            {"mlip_epa": -4.0, "ref_epa": -4.0},
            {"mlip_epa": -3.0, "ref_epa": -5.0},
        ]
        m = compute_ranking_metrics(results, stable_only=False)
        assert m["kendall_tau"] == pytest.approx(-1.0)
        assert m["n_structures"] == 3

    def test_mae_calculation(self):
        results = [
            {"mlip_epa": -5.0, "ref_epa": -5.1},
            {"mlip_epa": -4.0, "ref_epa": -3.8},
        ]
        m = compute_ranking_metrics(results, stable_only=False)
        assert m["mae_epa"] == pytest.approx(0.15)

    def test_insufficient_data_returns_none(self):
        m = compute_ranking_metrics([], stable_only=False)
        assert m["kendall_tau"] is None
        assert m["p_value"] is None
        assert m["mae_epa"] is None
        assert m["n_structures"] == 0

    def test_single_structure_returns_none(self):
        results = [{"mlip_epa": -5.0, "ref_epa": -5.0}]
        m = compute_ranking_metrics(results, stable_only=False)
        assert m["kendall_tau"] is None
        assert m["n_structures"] == 1

    def test_stable_only_filters_when_phonon_data_present(self):
        results = [
            {
                "mlip_epa": -5.0,
                "ref_epa": -5.0,
                "mlip_phonon_freq": 1.0,
                "ref_phonon_freq": 1.0,
            },
            {
                "mlip_epa": -4.0,
                "ref_epa": -4.0,
                "mlip_phonon_freq": -1.0,
                "ref_phonon_freq": 1.0,
            },
            {
                "mlip_epa": -3.0,
                "ref_epa": -3.0,
                "mlip_phonon_freq": 1.0,
                "ref_phonon_freq": -1.0,
            },
            {
                "mlip_epa": -2.0,
                "ref_epa": -2.0,
                "mlip_phonon_freq": 1.0,
                "ref_phonon_freq": 1.0,
            },
        ]
        m = compute_ranking_metrics(results, phonon_cutoff=-0.15, stable_only=True)
        assert m["stable_only_applied"] is True
        assert m["n_structures"] == 2  # only -5.0 and -2.0 are stable by both

    def test_stable_only_skipped_without_phonon_data(self):
        results = [
            {"mlip_epa": -5.0, "ref_epa": -5.0},
            {"mlip_epa": -4.0, "ref_epa": -4.0},
        ]
        m = compute_ranking_metrics(results, stable_only=True)
        assert m["stable_only_applied"] is False
        assert m["n_structures"] == 2

    def test_stable_only_skipped_with_partial_phonon_data(self):
        results = [
            {
                "mlip_epa": -5.0,
                "ref_epa": -5.0,
                "mlip_phonon_freq": 1.0,
                "ref_phonon_freq": None,
            },
            {
                "mlip_epa": -4.0,
                "ref_epa": -4.0,
                "mlip_phonon_freq": 1.0,
                "ref_phonon_freq": 1.0,
            },
        ]
        m = compute_ranking_metrics(results, stable_only=True)
        assert m["stable_only_applied"] is False
        assert m["n_structures"] == 2

    def test_all_filtered_out_returns_none(self):
        results = [
            {
                "mlip_epa": -5.0,
                "ref_epa": -5.0,
                "mlip_phonon_freq": -1.0,
                "ref_phonon_freq": -1.0,
            },
            {
                "mlip_epa": -4.0,
                "ref_epa": -4.0,
                "mlip_phonon_freq": -1.0,
                "ref_phonon_freq": -1.0,
            },
        ]
        m = compute_ranking_metrics(results, phonon_cutoff=-0.15, stable_only=True)
        assert m["stable_only_applied"] is True
        assert m["kendall_tau"] is None
        assert m["n_structures"] == 0


# ------------------------------------------------------------------ #
#  compute_stability_metrics
# ------------------------------------------------------------------ #


class TestStabilityMetrics:
    def test_perfect_classification(self):
        results = [
            {"mlip_phonon_freq": 1.0, "ref_phonon_freq": 1.0},
            {"mlip_phonon_freq": -1.0, "ref_phonon_freq": -1.0},
        ]
        m = compute_stability_metrics(results, phonon_cutoff=-0.15)
        assert m is not None
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)
        assert m["n_total"] == 2
        assert m["n_stable_ref"] == 1
        assert m["n_stable_mlip"] == 1

    def test_all_false_positives(self):
        results = [
            {"mlip_phonon_freq": 1.0, "ref_phonon_freq": -1.0},
            {"mlip_phonon_freq": 1.0, "ref_phonon_freq": -1.0},
        ]
        m = compute_stability_metrics(results, phonon_cutoff=-0.15)
        assert m is not None
        assert m["precision"] == pytest.approx(0.0)
        assert m["recall"] == pytest.approx(0.0)
        assert m["f1"] == pytest.approx(0.0)

    def test_all_false_negatives(self):
        results = [
            {"mlip_phonon_freq": -1.0, "ref_phonon_freq": 1.0},
            {"mlip_phonon_freq": -1.0, "ref_phonon_freq": 1.0},
        ]
        m = compute_stability_metrics(results, phonon_cutoff=-0.15)
        assert m is not None
        assert m["precision"] == pytest.approx(0.0)
        assert m["recall"] == pytest.approx(0.0)

    def test_mixed_classification(self):
        results = [
            {"mlip_phonon_freq": 1.0, "ref_phonon_freq": 1.0},  # TP
            {"mlip_phonon_freq": 1.0, "ref_phonon_freq": -1.0},  # FP
            {"mlip_phonon_freq": -1.0, "ref_phonon_freq": 1.0},  # FN
            {"mlip_phonon_freq": -1.0, "ref_phonon_freq": -1.0},  # TN
        ]
        m = compute_stability_metrics(results, phonon_cutoff=-0.15)
        assert m is not None
        assert m["precision"] == pytest.approx(0.5)  # 1/(1+1)
        assert m["recall"] == pytest.approx(0.5)  # 1/(1+1)
        assert m["f1"] == pytest.approx(0.5)  # 2*0.5*0.5/(0.5+0.5)
        assert m["n_stable_ref"] == 2
        assert m["n_stable_mlip"] == 2

    def test_returns_none_for_empty(self):
        assert compute_stability_metrics([], phonon_cutoff=-0.15) is None

    def test_returns_none_for_missing_data(self):
        results = [
            {"mlip_phonon_freq": None, "ref_phonon_freq": 1.0},
            {"mlip_phonon_freq": 1.0, "ref_phonon_freq": None},
        ]
        assert compute_stability_metrics(results, phonon_cutoff=-0.15) is None

    def test_partial_data_uses_valid_only(self):
        results = [
            {"mlip_phonon_freq": 1.0, "ref_phonon_freq": 1.0},
            {"mlip_phonon_freq": None, "ref_phonon_freq": 1.0},
        ]
        m = compute_stability_metrics(results, phonon_cutoff=-0.15)
        assert m is not None
        assert m["n_total"] == 1

    def test_custom_cutoff(self):
        results = [
            {"mlip_phonon_freq": -0.2, "ref_phonon_freq": -0.2},
        ]
        m_loose = compute_stability_metrics(results, phonon_cutoff=-0.5)
        assert m_loose is not None
        assert m_loose["n_stable_ref"] == 1

        m_strict = compute_stability_metrics(results, phonon_cutoff=-0.15)
        assert m_strict is not None
        assert m_strict["n_stable_ref"] == 0
