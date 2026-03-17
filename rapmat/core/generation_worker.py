"""Lightweight generation worker for ProcessPoolExecutor.

This module is intentionally kept free of heavy imports (torch, MatterSim,
CUDA-backed packages) so that subprocess workers on Windows (which use the
``spawn`` start method) do not duplicate GPU contexts or load MLIP models.
"""

_worker_descriptor = None


def init_generation_worker(
    species: list,
    r_cut: float,
    n_max: int,
    l_max: int,
    periodic: bool,
) -> None:
    """Process initializer: create SOAPDescriptor in worker process."""
    global _worker_descriptor
    from rapmat.storage.descriptors import SOAPDescriptor

    _worker_descriptor = SOAPDescriptor(
        species=species, r_cut=r_cut, n_max=n_max, l_max=l_max, periodic=periodic
    )


def generate_one_structure(
    struct_id: str,
    spg: int,
    fu: int,
    elements: list,
    formula_values: list,
    search_dim: int,
    thickness_cutoff: float | None,
) -> tuple:
    """Pure CPU work: pyxtal generation + SOAP compute.

    Returns ``(status, struct_id, atoms, vector)``.
    """
    import pyxtal

    global _worker_descriptor

    elements_number = [n * fu for n in formula_values]
    try:
        crystal = pyxtal.pyxtal()
        crystal.from_random(
            dim=search_dim,
            group=spg,
            species=elements,
            numIons=elements_number,
            max_count=1000,
            thickness=thickness_cutoff if search_dim == 2 else None,
        )
        if crystal.valid:
            atoms = crystal.to_ase()
            vec = (
                _worker_descriptor.compute(atoms)
                if _worker_descriptor is not None
                else None
            )
            return ("generated", struct_id, atoms, vec)
        return ("discarded", struct_id, None, None)
    except pyxtal.msg.Comp_CompatibilityError:
        return ("discarded", struct_id, None, None)
    except RuntimeError:
        return ("error", struct_id, None, None)
