import warnings
import numpy as np

from typing import Optional, Tuple, Type

from ase import Atoms
from ase.filters import Filter, FrechetCellFilter
from ase.optimize import BFGS


def _max_force(atoms) -> float:
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
