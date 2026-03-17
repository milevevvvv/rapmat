import warnings
from typing import Optional, Tuple, Type

import numpy as np
from ase import Atoms
from ase.filters import Filter, FrechetCellFilter
from ase.optimize import BFGS



def _max_force(atoms) -> float:
    """Maximum per-atom force norm."""
    return float(np.max(np.linalg.norm(atoms.get_forces(), axis=1)))


def structure_relax(
    atoms: Atoms,
    force_conv_crit: float = 5e-2,
    steps_max: int = 500,
    opt_logfile: Optional[str] = None,
    mask: Optional[np.ndarray] = None,
    filter_cls: Type[Filter] = FrechetCellFilter,
    forces_break: float = None,
    optimizer_cls: Type = BFGS,
    cleanup_gpu: bool = True,
    suppress_warnings: bool = True,
    scalar_pressure: float = 0.0,
    cancel_flag: list[bool] | None = None,
    progress_callback=None,
) -> Tuple[bool, Atoms]:
    """Relax an atomic structure using an ASE optimizer with a cell filter.

    Iteratively minimises forces on ``atoms`` until either
    ``force_conv_crit`` is satisfied, ``steps_max`` is reached, or the
    maximum per-atom force exceeds ``forces_break`` (early abort).

    Args:
        atoms: Structure to relax.  Must have a calculator attached.
        force_conv_crit: Force convergence criterion in eV/Å.
        steps_max: Maximum number of optimisation steps.
        opt_logfile: Path for the optimizer log file (``None`` = silent).
        mask: Voigt-notation boolean mask forwarded to *filter*
            (e.g. ``[1,1,0,0,0,1]`` to fix the *z*-cell vector).
        filter_cls: Cell-filter class applied before optimisation.
        forces_break: If the max per-atom force exceeds this value during
            optimisation the run is aborted and ``(False, atoms)`` is
            returned immediately.
        optimizer_cls: ASE optimizer class to use (default :class:`BFGS`).
        cancel_flag: A 1-element list containing a boolean. If it becomes
            ``[True]`` during optimisation, the run is aborted early.
        progress_callback: A callable invoked as
            ``progress_callback(step: int, max_steps: int, msg: str)``
            per iteration.

    Returns:
        A ``(converged, relaxed_atoms)`` tuple.

    Raises:
        RuntimeError: If no calculator is set.
    """
    import torch

    if atoms.calc is None:
        raise RuntimeError("No calculator set for the structure.")


    atoms_cf = filter_cls(atoms, mask=mask, scalar_pressure=scalar_pressure)

    last_fmax = float("inf")
    force_broken = False
    ctx = warnings.catch_warnings() if suppress_warnings else None
    step = 0
    try:
        if ctx is not None:
            ctx.__enter__()
            warnings.simplefilter("ignore")
        with optimizer_cls(atoms_cf, logfile=opt_logfile) as optimizer:
            iterator = optimizer.irun(fmax=force_conv_crit, steps=steps_max)

            for _ in iterator:
                if cancel_flag and cancel_flag[0]:
                    force_broken = True
                    break
                step += 1
                last_fmax = _max_force(atoms)

                if progress_callback is not None:
                    progress_callback(
                        step,
                        steps_max,
                        f"Step {step}/{steps_max} (fmax={last_fmax:.4f})",
                    )

                if forces_break is not None and last_fmax > forces_break:
                    force_broken = True
                    break

    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)

    converged = not force_broken and _max_force(atoms_cf) <= force_conv_crit

    if cleanup_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return converged, atoms_cf.atoms
