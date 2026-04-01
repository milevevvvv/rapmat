from enum import Enum
from pathlib import Path


class Domain(str, Enum):
    BULK = "bulk"
    MONOLAYER = "monolayer"


# ------------------------------------------------------------------ #
#  Orchestration loops (used by TUI and tests)
# ------------------------------------------------------------------ #


from rapmat.core.generation_worker import (
    generate_one_structure as _generate_one_structure,
    init_generation_worker as _init_generation_worker,
)


def run_processing_loop(
    run_name: str,
    store,
    config: dict,
    workdir_path: Path,
    descriptor,
    worker_id: str | None = None,
    progress_callback=None,
    cancel_flag: list[bool] | None = None,
):
    """Common processing loop for running relaxations and filtering.

    """
    import numpy as np
    import torch
    from ase.units import GPa as _GPa
    from rapmat.calculators.factory import load_calculator
    from rapmat.calculators import CalculatorCallback, Calculators
    from rapmat.core.dedup import confirm_duplicates
    from rapmat.core.relaxation import structure_relax
    from rapmat.core.sanity import check_sanity
    from rapmat.utils.structure import calculate_thickness, format_spg, standardize_atoms
    from rapmat.utils.console import console, err_console

    class _ProgressCalcCallback:
        """Adapter: forwards calculator status to the TUI progress callback."""
        def on_status(self, message: str) -> None:
            if progress_callback:
                progress_callback(0, 0, message)

    _calc_cb = _ProgressCalcCallback()

    calculator_name = config.get("calculator", "MATTERSIM").upper()
    calculator_config = config.get("calculator_config", {})
    domain_val = config.get("domain", "bulk")
    search_dim = 3 if domain_val == "bulk" else 2
    skip_not_converged = config.get("skip_not_converged", False)
    use_dedup = config.get("dedup", False)
    dedup_threshold = config.get("dedup_threshold", 1e-2)
    symprec = config.get("symprec", 1e-5)
    pressure_gpa = config.get("pressure_gpa", 0.0)
    pressure_evA3 = pressure_gpa * _GPa
    use_pymatgen_dedup = config.get("pymatgen_dedup", False)
    pymatgen_ltol = config.get("pymatgen_ltol", 0.2)
    pymatgen_stol = config.get("pymatgen_stol", 0.3)
    pymatgen_angle_tol = config.get("pymatgen_angle_tol", 5.0)
    use_force_dedup = config.get("force_dedup", False)
    force_cosine_threshold = config.get("force_cosine_threshold", 0.95)
    min_dist = config.get("min_dist", 0.5)
    use_sanity_pymatgen = config.get("sanity_pymatgen", False)
    sanity_pymatgen_tol = config.get("sanity_pymatgen_tol", 0.5)

    force_conv_crit = float(config.get("force_conv_crit", 5e-2))
    steps_max = int(config.get("steps_max", 500))
    forces_break = config.get("forces_break", 1000.0)
    if forces_break is not None:
        forces_break = float(forces_break)

    calculator_workdir_path = workdir_path / Path("calculator")
    calculator_workdir_path.mkdir(parents=True, exist_ok=True)

    candidates = store.get_unrelaxed_candidates(run_name)

    relaxed_structures = []

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    _calc_cb.on_status(f"Loading calculator {calculator_name}...")
    calculator = load_calculator(
        Calculators(calculator_name), config=calculator_config,
        callback=_calc_cb,
    )

    counter: int = 0
    discarded_conv = 0
    discarded_sanity = 0
    discarded_unstable = 0
    discarded_dup = 0
    discarded_candidate_dup = 0
    n_candidates = len(candidates)

    def _report(msg: str) -> None:
        if progress_callback:
            progress_callback(counter, n_candidates, msg)

    def _run_loop():
        nonlocal counter, discarded_conv, discarded_sanity
        nonlocal discarded_unstable, discarded_dup, discarded_candidate_dup
        nonlocal calculator

        for candidate in candidates:
            counter += 1
            if worker_id and counter % 10 == 0:
                store.update_heartbeat(run_name, worker_id)
            struct_id = candidate["id"]

            for attempt in range(3):
                structure = candidate["atoms"].copy()
                try:
                    structure.calc = calculator
                    structure.info["initial_spg"] = format_spg(
                        structure, symprec=symprec
                    )

                    candidate_vec = descriptor.compute(structure)
                    candidate_forces = (
                        structure.get_forces() if use_force_dedup else None
                    )

                    if candidate_forces is not None:
                        structure.info["initial_forces"] = candidate_forces.tolist()

                    store.update_generated_structure(
                        struct_id, structure, vector=candidate_vec
                    )

                    if use_dedup:
                        _report("Requesting candidate neighbours...")
                        nearby_candidates = store.get_nearby_structures(
                            candidate_vec,
                            dedup_threshold,
                            run_id=run_name,
                            statuses=("generated", "relaxed"),
                            exclude_ids=[struct_id],
                        )
                        if nearby_candidates:
                            _report("Confirming candidate duplicates...")
                            candidate_dup_energy = confirm_duplicates(
                                structure,
                                nearby_candidates,
                                use_pymatgen=use_pymatgen_dedup,
                                ltol=pymatgen_ltol,
                                stol=pymatgen_stol,
                                angle_tol=pymatgen_angle_tol,
                                use_forces=use_force_dedup,
                                candidate_forces=candidate_forces,
                                force_cosine_threshold=force_cosine_threshold,
                            )
                            if candidate_dup_energy is not None:
                                discarded_candidate_dup += 1
                                _report(f"Discarded {struct_id}: candidate duplicate")
                                store.update_structure(
                                    struct_id, status="discarded", vector=candidate_vec
                                )
                                break

                    _report("Relaxing...")

                    def _optim_cb(step: int, max_steps: int, msg: str) -> None:
                        if progress_callback:
                            msg_fmt = f"Relaxing {struct_id}: {msg}"
                            progress_callback(counter, n_candidates, msg_fmt, False)

                    converged, relaxed_structure = structure_relax(
                        structure,
                        force_conv_crit=force_conv_crit,
                        steps_max=steps_max,
                        mask=[1, 1, 0, 0, 0, 1] if domain_val == "monolayer" else None,
                        opt_logfile=str(
                            calculator_workdir_path
                            / Path(f"opt_{struct_id.replace('/', '_')}.log")
                        ),
                        scalar_pressure=pressure_evA3,
                        forces_break=forces_break,
                        cancel_flag=cancel_flag,
                        progress_callback=_optim_cb,
                    )

                    if cancel_flag and cancel_flag[0]:
                        break

                    _report("Data collection...")
                    relaxed_structure.info["energy"] = (
                        relaxed_structure.get_potential_energy()
                    )
                    forces = relaxed_structure.get_forces()
                    relaxed_structure.info["forces"] = forces
                    relaxed_structure.info["fmax"] = np.max(
                        np.linalg.norm(forces, axis=1)
                    )
                    relaxed_structure.info["converged"] = converged
                    relaxed_structure.info["initial_spg"] = structure.info[
                        "initial_spg"
                    ]
                    
                    relaxed_structure.info["final_spg"] = format_spg(
                        relaxed_structure, symprec=symprec
                    )

                    _report("Metadata preparation...")
                    energy = relaxed_structure.info["energy"]
                    volume = relaxed_structure.get_volume()
                    enthalpy = energy + pressure_evA3 * volume

                    meta = {
                        "energy_per_atom": energy / len(relaxed_structure),
                        "energy_total": energy,
                        "enthalpy_per_atom": enthalpy / len(relaxed_structure),
                        "volume": volume,
                        "fmax": relaxed_structure.info["fmax"],
                        "converged": relaxed_structure.info["converged"],
                        "thickness": 0.0,
                    }

                    _report("Filtering...")

                    if skip_not_converged and not converged:
                        discarded_conv += 1
                        _report(
                            f"Discarded {struct_id}: not converged (fmax={meta['fmax']:.4f})"
                        )
                        store.update_structure(
                            struct_id,
                            status="discarded",
                            atoms=relaxed_structure,
                            metadata=meta,
                        )
                        break

                    if not check_sanity(
                        relaxed_structure,
                        min_dist=min_dist,
                        use_pymatgen=use_sanity_pymatgen,
                        pymatgen_tol=sanity_pymatgen_tol,
                    ):
                        discarded_sanity += 1
                        _report(f"Discarded {struct_id}: failed sanity check")
                        store.update_structure(
                            struct_id,
                            status="discarded",
                            atoms=relaxed_structure,
                            metadata=meta,
                        )
                        break

                    if search_dim == 2:
                        current_thickness = calculate_thickness(relaxed_structure)
                        relaxed_structure.info["thickness"] = current_thickness
                        meta["thickness"] = current_thickness

                    if use_dedup:
                        _report("Computing relaxed vector...")
                        vec = descriptor.compute(relaxed_structure)

                        _report("Requesting relaxed neighbours...")
                        nearby = store.get_nearby_structures(
                            vec, dedup_threshold, run_id=run_name, statuses=("relaxed",)
                        )

                        _report("Confirming relaxed duplicates...")
                        min_energy = confirm_duplicates(
                            relaxed_structure,
                            nearby,
                            use_pymatgen=use_pymatgen_dedup,
                            ltol=pymatgen_ltol,
                            stol=pymatgen_stol,
                            angle_tol=pymatgen_angle_tol,
                            use_forces=use_force_dedup,
                            candidate_forces=(
                                relaxed_structure.get_forces()
                                if use_force_dedup
                                else None
                            ),
                            force_cosine_threshold=force_cosine_threshold,
                        )
                    else:
                        vec = descriptor.compute(relaxed_structure)
                        min_energy = None
                    if min_energy is not None and meta["energy_per_atom"] > min_energy:
                        discarded_dup += 1
                        _report(
                            f"Discarded {struct_id}: relaxed duplicate (E={meta['energy_per_atom']:.4f} > {min_energy:.4f})"
                        )
                        store.update_structure(
                            struct_id,
                            status="discarded",
                            atoms=relaxed_structure,
                            vector=vec,
                            metadata=meta,
                        )
                        break

                    _report("Saving to database...")
                    store.update_structure(
                        struct_id,
                        status="relaxed",
                        atoms=relaxed_structure,
                        vector=vec,
                        metadata=meta,
                    )
                    relaxed_structures.append(relaxed_structure)
                    break

                except Exception as ex:
                    _report(f"ERROR: {ex}")
                    if attempt == 2:
                        err_console.print(
                            f"[red]Failed to relax structure {struct_id}: {ex}[/red]"
                        )
                        store.update_structure(struct_id, status="error")
                        break

                    try:
                        del calculator
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    
                    _report(f"Reloading calculator {calculator_name} after error...")
                    try:
                        calculator = load_calculator(
                            Calculators(calculator_name), config=calculator_config,
                            callback=_calc_cb,
                        )
                    except Exception as reload_ex:
                        err_console.print(f"[red]Calculator reload failed: {reload_ex}[/red]")
                        _report(f"CRITICAL ERROR: Reload failed: {reload_ex}")
                        store.update_structure(struct_id, status="error")
                        break # Give up on this structure if reload fails

            if progress_callback:
                progress_callback(
                    counter, n_candidates, f"Processed {counter}/{n_candidates}"
                )

    _run_loop()

    n_relaxed = len(relaxed_structures)
    discarded_parts = [
        f"{discarded_candidate_dup} cand-dup",
        f"{discarded_conv} conv",
        f"{discarded_sanity} sanity",
        f"{discarded_unstable} unstable",
        f"{discarded_dup} dup",
    ]
    discarded_str = ", ".join(discarded_parts)

    pressure_msg = f" | Pressure: {pressure_gpa} GPa" if pressure_gpa > 0 else ""
    console.print(
        f"\n[bold green]Done.[/bold green] Run: [bold]{run_name}[/bold] | "
        f"Storage: {store._db_url}{pressure_msg}\n"
        f"Relaxed: [bold]{n_relaxed}[/bold] | Discarded: {discarded_str}\n"
    )

    return None


def run_generation_loop(
    run_name: str,
    store,
    config: dict,
    worker_id: str | None = None,
    descriptor=None,
    workers: int = 1,
    progress_callback=None,
) -> int:
    """Generate structures from DB placeholders (status='generating').

    Returns the number of successfully generated structures.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from rapmat.utils.console import console, err_console

    domain_val = config.get("domain", "bulk")
    search_dim = 3 if domain_val == "bulk" else 2
    formula = config.get("formula", {})
    thickness_cutoff = config.get("thickness_cutoff", None)
    run_seed = config.get("seed")  # int | None

    elements = list(formula.keys())
    formula_values = list(formula.values())

    placeholders = store.get_pending_generation(run_name)
    if not placeholders:
        if progress_callback is None:
            console.print("[green]No structures left to generate.[/green]")
        return 0

    n_placeholders = len(placeholders)
    if progress_callback is None:
        console.print(
            f"[bold cyan]Generating structures for run: {run_name}...[/bold cyan]\n"
            f"Found [bold]{n_placeholders}[/bold] placeholders to generate."
        )

    generated = 0
    discarded = 0
    errors = 0

    def _advance(counter: int) -> None:
        if progress_callback:
            progress_callback(
                counter,
                n_placeholders,
                f"Generating {counter}/{n_placeholders}",
            )

    def _handle_result(status, struct_id, atoms, vec, spg, fu):
        nonlocal generated, discarded, errors
        match status:
            case "generated":
                store.update_generated_structure(struct_id, atoms, vector=vec)
                generated += 1
            case "discarded":
                store.discard_generation_placeholder(struct_id)
                discarded += 1
            case "error":
                err_console.print(f"[red]Structure for group {spg} / fu {fu} failed[/red]")
                store.discard_generation_placeholder(struct_id)
                errors += 1

    if workers <= 1:
        import rapmat.core.generation_worker as _gw
        _gw._worker_descriptor = descriptor

        for counter, ph in enumerate(placeholders, start=1):
            if worker_id and counter % 20 == 0:
                store.update_heartbeat(run_name, worker_id)

            spg = ph["gen_spg"]
            fu = ph["gen_fu"]
            struct_seed = (
                (run_seed + counter) % (2**32) if run_seed is not None else None
            )
            status, struct_id, atoms, vec = _generate_one_structure(
                ph["id"], spg, fu, elements, formula_values, search_dim, None,
                seed=struct_seed,
            )
            _handle_result(status, struct_id, atoms, vec, spg, fu)
            _advance(counter)

    else:
        if descriptor is not None:
            initargs = (
                elements,
                descriptor._r_cut,
                descriptor._n_max,
                descriptor._l_max,
                descriptor._periodic,
            )
            pool = ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_generation_worker,
                initargs=initargs,
            )
        else:
            pool = ProcessPoolExecutor(max_workers=workers)

        with pool:
            futures = {
                pool.submit(
                    _generate_one_structure,
                    ph["id"],
                    ph["gen_spg"],
                    ph["gen_fu"],
                    elements,
                    formula_values,
                    search_dim,
                    None,
                    seed=(
                        (run_seed + idx) % (2**32)
                        if run_seed is not None
                        else None
                    ),
                ): ph
                for idx, ph in enumerate(placeholders, start=1)
            }

            for counter, future in enumerate(as_completed(futures), start=1):
                if worker_id and counter % 20 == 0:
                    store.update_heartbeat(run_name, worker_id)

                status, struct_id, atoms, vec = future.result()
                ph = futures[future]
                _handle_result(status, struct_id, atoms, vec, ph["gen_spg"], ph["gen_fu"])
                _advance(counter)

    return generated
