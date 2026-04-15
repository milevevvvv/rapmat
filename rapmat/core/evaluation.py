from typing import Sequence
from rapmat.utils.structure import standardize_atoms

# ------------------------------------------------------------------ #
#  Evaluation loop (used by TUI and tests)
# ------------------------------------------------------------------ #


def run_eval_loop(
    pending: list[dict],
    store,
    run_name: str,
    calculator,
    calculator_name: str,
    config_json: str,
    *,
    run_phonons: bool = False,
    phonon_displacement: float = 1e-2,
    phonon_supercell: tuple = (3, 3, 3),
    phonon_mesh: tuple = (20, 20, 20),
    progress_callback=None,
    log_callback=None,
    reduce_to_primitive: bool = True,
    symprec: float = 1e-3,
) -> None:
    from rapmat.core.phonon import get_mesh_min_frequency, structure_calculate_phonons
    from rapmat.utils.console import err_console

    from rapmat.calculators import cleanup_calculator_files

    n_total = len(pending)
    for i, rec in enumerate(pending, 1):
        atoms = rec["atoms"].copy()
        atoms.pbc = True  # NOTE: ensure full PBC for calculators like VASP, dirty fix, limit to VASP later

        cleanup_calculator_files(calculator)

        atoms.calc = calculator

        try:
            ref_energy = atoms.get_potential_energy()
            ref_epa = ref_energy / len(atoms)

            ref_phonon_freq = None
            if run_phonons:
                if reduce_to_primitive:
                    atoms_len_before = len(atoms)
                    atoms = standardize_atoms(atoms, to_primitive=True, symprec=symprec)
                    atoms_len_after = len(atoms)

                    atoms.calc = calculator

                    if log_callback:
                        log_callback(
                            f"Reducing {rec['id']}: {atoms_len_before} -> {atoms_len_after} atoms"
                        )

                phonons = structure_calculate_phonons(
                    atoms,
                    displacement=phonon_displacement,
                    supercell=phonon_supercell,
                    qpoint_mesh=phonon_mesh,
                    progress_callback=progress_callback,
                )
                ref_phonon_freq = get_mesh_min_frequency(phonons)

            store.add_evaluation(
                structure_id=rec["id"],
                run_name=run_name,
                calculator=calculator_name,
                config_json=config_json,
                energy_per_atom=ref_epa,
                energy_total=ref_energy,
                min_phonon_freq=ref_phonon_freq,
            )
        except Exception as e:
            import traceback

            err_msg = f"Failed to evaluate structure {rec['id']}: {e}"
            if log_callback:
                log_callback(err_msg)
                log_callback(traceback.format_exc())
            else:
                err_console.print(f"[red]{err_msg}[/red]")

        if progress_callback:
            progress_callback(i, n_total, f"Evaluated {i}/{n_total}")


# ------------------------------------------------------------------ #
#  Pure metric helpers
# ------------------------------------------------------------------ #


def compute_ranking_metrics(
    results: Sequence[dict],
    phonon_cutoff: float = -0.15,
    stable_only: bool = True,
) -> dict:
    from scipy.stats import kendalltau

    subset = list(results)
    stable_only_applied = False

    if stable_only:
        has_phonon = all(
            r.get("mlip_phonon_freq") is not None
            and r.get("ref_phonon_freq") is not None
            for r in subset
        )
        if has_phonon:
            subset = [
                r
                for r in subset
                if r["mlip_phonon_freq"] >= phonon_cutoff
                and r["ref_phonon_freq"] >= phonon_cutoff
            ]
            stable_only_applied = True

    n = len(subset)
    if n < 2:
        return {
            "kendall_tau": None,
            "p_value": None,
            "mae_epa": None,
            "n_structures": n,
            "stable_only_applied": stable_only_applied,
        }

    mlip_vals = [r["mlip_epa"] for r in subset]
    ref_vals = [r["ref_epa"] for r in subset]

    tau, p_value = kendalltau(mlip_vals, ref_vals)
    mae = sum(abs(m - r) for m, r in zip(mlip_vals, ref_vals)) / n

    return {
        "kendall_tau": float(tau),
        "p_value": float(p_value),
        "mae_epa": float(mae),
        "n_structures": n,
        "stable_only_applied": stable_only_applied,
    }


def compute_stability_metrics(
    results: Sequence[dict],
    phonon_cutoff: float = -0.15,
) -> dict | None:
    valid = [
        r
        for r in results
        if r.get("mlip_phonon_freq") is not None
        and r.get("ref_phonon_freq") is not None
    ]
    if not valid:
        return None

    tp = fp = fn = tn = 0
    for r in valid:
        ref_stable = r["ref_phonon_freq"] >= phonon_cutoff
        mlip_stable = r["mlip_phonon_freq"] >= phonon_cutoff
        if ref_stable and mlip_stable:
            tp += 1
        elif not ref_stable and mlip_stable:
            fp += 1
        elif ref_stable and not mlip_stable:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_total": len(valid),
        "n_stable_ref": tp + fn,
        "n_stable_mlip": tp + fp,
    }
