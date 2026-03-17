"""Pytest configuration and fixtures."""

# Set spglib error handling mode before any imports to avoid deprecation warnings
import spglib

spglib.OLD_ERROR_HANDLING = False

import sys
import torch

if sys.platform == "win32":
    sys.modules.pop("urwid.display.curses", None)

from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
from rapmat.storage import SurrealDBStore
from rapmat.storage.base import StructureStore

VECTOR_DIM = 10
_TEST_DESC_ID = "test000000000000000000000000000000000000000000000000000000000000"


def _ensure_descriptor(store: StructureStore) -> None:
    """Register the test descriptor if not already active."""
    store.register_descriptor(_TEST_DESC_ID, VECTOR_DIM)


def add_relaxed_structure(
    store: StructureStore,
    run_name: str,
    atoms: Atoms,
    energy_per_atom: float,
    struct_id: str,
    vector: np.ndarray | None = None,
) -> None:
    """Add a candidate and immediately mark it relaxed with the given energy."""
    n_atoms = len(atoms)
    energy_total = energy_per_atom * n_atoms
    if vector is None:
        vector = np.zeros(VECTOR_DIM, dtype=np.float32)
    _ensure_descriptor(store)
    store.add_candidate(atoms, vector, run_name, struct_id)
    store.update_structure(
        struct_id,
        "relaxed",
        atoms=atoms,
        vector=vector,
        metadata={
            "energy_per_atom": energy_per_atom,
            "energy_total": energy_total,
            "fmax": 0.01,
            "converged": True,
        },
    )


# ------------------------------------------------------------------ #
#  Common atom fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def cu_fcc():
    return bulk("Cu", "fcc", a=3.615)


@pytest.fixture
def cu_bcc():
    return bulk("Cu", "bcc", a=2.87)


@pytest.fixture
def al_fcc():
    return bulk("Al", "fcc", a=4.05)


# ------------------------------------------------------------------ #
#  Store factory fixture (dual-backend)
# ------------------------------------------------------------------ #


def _make_surreal(tmp_path):
    store = SurrealDBStore.from_path(tmp_path / "surreal_db")
    _ensure_descriptor(store)
    return store


@pytest.fixture(params=["surreal"])
def any_store(request, tmp_path):
    """Parametrized fixture yielding a StructureStore for each backend."""
    if request.param == "surreal":
        return _make_surreal(tmp_path)
    raise ValueError(f"Unknown backend: {request.param}")


@pytest.fixture
def surreal_store(tmp_path):
    """SurrealDB-only store fixture."""
    return _make_surreal(tmp_path)


# ------------------------------------------------------------------ #
#  Pre-populated hull store
# ------------------------------------------------------------------ #


@pytest.fixture
def hull_store(tmp_path):
    """SurrealDBStore pre-populated with a synthetic Al-Cu binary system.

    Reference energies (eV/atom):
        Al = -3.0,  Cu = -4.0

    Intermediate structures:
        AlCu   (epa=-5.0, 2 atoms)  -> formation_energy = -1.5  (ON hull)
        Al3Cu  (epa=-3.3, 4 atoms)  -> formation_energy = -0.05 (ABOVE hull)
    """
    store = SurrealDBStore.from_path(tmp_path / "hull_db")

    store.create_study("test-study", system="Al-Cu", domain="bulk", calculator="mock")

    # Pure-Al endpoint
    store.create_run(
        "al-run",
        config={"formula": {"Al": 1}, "calculator": "mock"},
        study_id="test-study",
    )
    add_relaxed_structure(store, "al-run", bulk("Al", "fcc", a=4.05), -3.0, "al-run/1")

    # Pure-Cu endpoint
    store.create_run(
        "cu-run",
        config={"formula": {"Cu": 1}, "calculator": "mock"},
        study_id="test-study",
    )
    add_relaxed_structure(store, "cu-run", bulk("Cu", "fcc", a=3.615), -4.0, "cu-run/1")

    # AlCu intermediate    # AlCu ON hull
    store.create_run(
        "alcu-on",
        config={"formula": {"Al": 1, "Cu": 1}, "calculator": "mock"},
        study_id="test-study",
    )
    alcu = Atoms(
        symbols=["Al", "Cu"],
        positions=[[0, 0, 0], [1.5, 1.5, 1.5]],
        cell=[3, 3, 3],
        pbc=True,
    )
    add_relaxed_structure(store, "alcu-on", alcu, -5.0, "alcu-on/1")

    # Al3Cu intermediate (above hull)
    # epa = -3.0 gives formation_energy = (-12.0 - (-13.0)) / 4 = +0.25 eV/atom    # Al3Cu ABOVE hull
    store.create_run(
        "al3cu-off",
        config={"formula": {"Al": 3, "Cu": 1}, "calculator": "mock"},
        study_id="test-study",
    )
    al3cu = Atoms(
        symbols=["Al", "Al", "Al", "Cu"],
        positions=[[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0], [1.5, 1.5, 0]],
        cell=[3, 3, 3],
        pbc=True,
    )
    add_relaxed_structure(store, "al3cu-off", al3cu, -3.0, "al3cu-off/1")

    return store


# ------------------------------------------------------------------ #
#  CLI testing helpers
# ------------------------------------------------------------------ #


@contextmanager
def mock_pyxtal_generation(atoms_list=None):
    """Mock PyXtal generation for resumable-generation testing.

    Patches the ``pyxtal`` module so that ``run_generation_loop`` produces
    deterministic structures without requiring a real PyXtal install.

    Args:
        atoms_list: Optional list of ASE Atoms to cycle through.
            If None, returns Cu FCC (a=3.615) for every placeholder.
    """
    from unittest.mock import MagicMock

    if atoms_list is None:
        atoms_list = [bulk("Cu", "fcc", a=3.615)]

    _idx = [0]

    class _MockPyxtal:
        def __init__(self):
            self.valid = True

        def from_random(self, **kwargs):
            pass

        def to_ase(self):
            atoms = atoms_list[_idx[0] % len(atoms_list)]
            _idx[0] += 1
            return atoms

    class _CompCompatError(Exception):
        pass

    mock_msg = MagicMock()
    mock_msg.Comp_CompatibilityError = _CompCompatError

    mock_module = MagicMock()
    mock_module.pyxtal = _MockPyxtal
    mock_module.msg = mock_msg

    with patch.dict("sys.modules", {"pyxtal": mock_module, "pyxtal.msg": mock_msg}):
        yield


@contextmanager
def mock_calculator_factory():
    """Mock calculator factory to return EMT instead of loading real calculators."""
    from rapmat.calculators.factory import load_calculator

    def mock_load(calculator_enum, config=None):
        return EMT()

    with patch("rapmat.calculators.factory.load_calculator", side_effect=mock_load):
        yield
