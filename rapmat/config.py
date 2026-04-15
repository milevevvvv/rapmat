import json
import tomllib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import platformdirs

from rapmat.calculators import Calculators

APP_NAME = "rapmat-materials"

APP_TMPDIR_SUFFIX = "rapmatmaterials"
DEFAULT_DB_DIR = ".surrealdb"

APP_CONFIG_DIR = Path(platformdirs.user_config_dir(APP_NAME))
APP_DATA_DIR = Path(platformdirs.user_data_dir(APP_NAME))


@dataclass
class CalculatorParams:
    calculator_name: Calculators = Calculators.MATTERSIM
    workdir: str | None = None
    calculator_config: str | None = None
    calc_opt: tuple[str, ...] | None = None


def _parse_calc_opt_value(raw: str) -> object:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def resolve_calculator_config(calc: CalculatorParams) -> dict:
    config: dict = {}

    if calc.calculator_config is not None:
        config_path = Path(calc.calculator_config)
        if not config_path.is_file():
            raise ValueError(f"Calculator config file not found: {config_path}")
        with open(config_path, "rb") as f:
            config = tomllib.load(f)

    if calc.calc_opt:
        for item in calc.calc_opt:
            key, sep, value = item.partition("=")
            if not sep:
                raise ValueError(
                    f"Invalid calc-opt format: '{item}'. Expected KEY=VALUE."
                )
            config[key.strip()] = _parse_calc_opt_value(value.strip())

    return config


@dataclass
class PhononParams:
    phonon_supercell: tuple[int, int, int] = (3, 3, 3)
    phonon_mesh: tuple[int, int, int] = (20, 20, 20)
    phonon_displacement: float = 1e-2
    phonon_cutoff: float = -0.15


@dataclass
class DedupParams:
    dedup: bool = False
    dedup_threshold: float = 1e-2
    pymatgen_dedup: bool = False
    pymatgen_ltol: float = 0.2
    pymatgen_stol: float = 0.3
    pymatgen_angle_tol: float = 5.0
    force_dedup: bool = False
    force_cosine_threshold: float = 0.95


@dataclass
class SanityParams:
    min_dist: float = 0.5
    sanity_pymatgen: bool = False
    sanity_pymatgen_tol: float = 0.5


@dataclass
class SymmetryParams:
    symprec: float = 1e-3


# ------------------------------------------------------------------ #
#  Database connection parameters
# ------------------------------------------------------------------ #


class DbMode(str, Enum):
    AUTO = "auto"
    FILE = "file"
    MEMORY = "memory"
    SERVER = "server"


class DbBackend(str, Enum):
    SURREAL = "surreal"


@dataclass
class DbParams:
    db_mode: DbMode = DbMode.AUTO
    db_path: str = str(APP_DATA_DIR / "surrealdb")
    db_backend: DbBackend = DbBackend.SURREAL
