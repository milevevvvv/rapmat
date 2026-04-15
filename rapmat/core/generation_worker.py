_worker_descriptor = None


def init_generation_worker(
    species: list,
    r_cut: float,
    n_max: int,
    l_max: int,
    periodic: bool,
) -> None:
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
    seed: int | None = None,
    max_count: int = 10,
) -> tuple:
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
            max_count=max_count,
            thickness=thickness_cutoff if search_dim == 2 else None,
            random_state=seed,
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
