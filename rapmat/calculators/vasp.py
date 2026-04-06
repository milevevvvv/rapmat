from pathlib import Path

from ase.calculators.vasp import Vasp


def build_calculator_vasp(config: dict, directory: Path | None = None) -> Vasp:
    """Build an ASE VASP calculator from a merged config dict.

    All key-value pairs in *config* are forwarded directly to
    ``ase.calculators.vasp.Vasp(**config)``.  The *directory* argument
    sets the VASP working directory (only if the config does not already
    contain one).
    """
    kwargs = dict(config)
    if directory is not None:
        kwargs["directory"] = str(directory)
    return Vasp(**kwargs)
