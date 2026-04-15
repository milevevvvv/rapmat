from pathlib import Path

from ase.calculators.vasp import Vasp


def build_calculator_vasp(config: dict, directory: Path | None = None) -> Vasp:
    kwargs = dict(config)
    if directory is not None:
        kwargs["directory"] = str(directory)

    if "txt" not in kwargs:
        workdir = kwargs.get("directory", ".")
        kwargs["txt"] = str(Path(workdir) / "vasp.out")

    return Vasp(**kwargs)
