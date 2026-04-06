from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import numpy as np
from ase import Atoms

# ------------------------------------------------------------------ #
#  Descriptor ABC
# ------------------------------------------------------------------ #


class StructureDescriptor(ABC):
    """Abstract base class for converting atomic structures to vector descriptors."""

    @abstractmethod
    def dimension(self) -> int:
        """Returns the dimension of the descriptor vector."""
        ...

    @abstractmethod
    def compute(self, atoms: Atoms) -> np.ndarray:
        """Computes the descriptor vector for the given structure.

        Returns a 1-D numpy array of floats.
        """
        ...

    @abstractmethod
    def code_version(self) -> str:
        """Returns the descriptor implementation's code version (e.g. '1.0')."""
        ...

    @abstractmethod
    def descriptor_id(self) -> str:
        """Returns a stable hash id for this descriptor configuration.

        The id is derived from descriptor name + params + descriptor code
        version, so that the same configuration yields the same id across runs.
        """
        ...

    def vec_col_name(self) -> str:
        """Short identifier derived from :meth:`descriptor_id`.

        Returns ``vec_`` followed by the first 12 hex characters of the
        descriptor id.
        """
        # TODO: move to storage backends
        return f"vec_{self.descriptor_id()[:12]}"


# ------------------------------------------------------------------ #
#  Store ABC
# ------------------------------------------------------------------ #


class StructureStore(ABC):
    """Backend-agnostic interface for persisting CSP structures and metadata."""

    # -- descriptor registration ----------------------------------------

    @abstractmethod
    def register_descriptor(
        self,
        desc_id: str,
        dim: int,
        meta: Optional[dict] = None,
    ) -> str:
        """Register a descriptor configuration.

        Ensures the backend is ready to store / search vectors of the given
        dimension.  Returns a short identifier for the descriptor space
        (e.g. ``vec_<hash>``).
        """
        # TODO: auto register on first use
        ...

    # -- run management -------------------------------------------------

    @abstractmethod
    def create_run(
        self,
        name: str,
        study_id: str,
        config: dict = {},
        worker_id: Optional[str] = None,
    ) -> str: ...

    @abstractmethod
    def get_run_metadata(self, name: str) -> Optional[dict]: ...

    @abstractmethod
    def update_run_config(self, name: str, config: dict) -> None:
        """Update the configuration dictionary for an existing run."""
        ...

    @abstractmethod
    def delete_run(self, run_name: str) -> None:
        """Permanently delete a run and all its associated structures."""
        ...

    @abstractmethod
    def list_runs(self) -> List[dict]: ...

    @abstractmethod
    def count_by_status(self, run_name: str) -> dict[str, int]: ...

    @abstractmethod
    def claim_run(self, run_name: str, worker_id: str) -> bool: ...

    @abstractmethod
    def release_run(self, run_name: str, final_status: str = "completed") -> None: ...

    @abstractmethod
    def update_heartbeat(self, run_name: str, worker_id: str) -> None: ...

    @abstractmethod
    def set_run_status(self, run_name: str, status: str) -> None: ...

    @abstractmethod
    def reclaim_stale_runs(self, timeout_minutes: int = 10) -> list[str]: ...

    # -- structures -----------------------------------------------------

    @abstractmethod
    def add_structures(self, run_name: str, structures: List[dict]) -> int:
        """Bulk-insert structure records.

        Each dict must contain at least ``"id"`` and ``"status"``.  Optional
        keys include ``"atoms"`` (ASE Atoms), ``"vector"`` (np.ndarray),
        ``"gen_spg"``, ``"gen_fu"``, ``"metadata"`` (dict with energy etc.),
        and ``"thickness"``.

        Returns the number of records inserted.
        """
        # TODO: move to objects with defaults instead of just dicts
        ...

    @abstractmethod
    def update_structure(self, struct_id: str, **fields) -> None:
        """Update fields on a single structure record.

        Supported keyword arguments mirror the structure columns:
        ``status``, ``atoms``, ``vector``, ``formula``, ``energy_per_atom``,
        ``energy_total``, ``fmax``, ``converged``, ``thickness``,
        ``enthalpy_per_atom``, ``volume``, ``min_phonon_freq``,
        ``metadata`` (convenience dict unpacked into the above scalars).
        """
        ...

    @abstractmethod
    def clear_run_phonon_results(self, run_name: str) -> None:
        """Reset ``min_phonon_freq`` to *None* for every structure in a run."""
        ...

    @abstractmethod
    def get_structures(
        self,
        run_name: str,
        *,
        status: Optional[str] = None,
        statuses: Optional[tuple[str, ...]] = None,
        fields: Optional[list[str]] = None,
        symprec: float = 1e-3,
    ) -> List[dict]:
        """Retrieve structures for a run, optionally filtered by status.

        Parameters
        ----------
        status
            Single status string filter (convenience shorthand).
        statuses
            Tuple of allowed statuses (takes precedence over *status*).
        fields
            Optional list of field names to return.  If ``None`` the backend
            returns its full default set (id, atoms, energy, status, ...).
        symprec
            Tolerance for space-group detection on the returned atoms.
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """Total number of structure records."""
        ...

    # -- vector search --------------------------------------------------

    @abstractmethod
    def find_neighbors(
        self,
        vector: np.ndarray,
        threshold: float,
        *,
        k: int = 500,
        run_id: Optional[str] = None,
        statuses: tuple[str, ...] = ("relaxed",),
        exclude_ids: Optional[List[str]] = None,
    ) -> List[dict]:
        """Return structures whose descriptor vector is within *threshold*.

        Each returned dict contains at least ``"id"``, ``"atoms"``,
        ``"energy_per_atom"``, ``"distance"``, and ``"forces"``.
        """
        ...

    # -- studies --------------------------------------------------------

    @abstractmethod
    def create_study(
        self,
        study_id: str,
        system: str,
        domain: str,
        calculator: str,
        config: Optional[dict] = None,
    ) -> str: ...

    @abstractmethod
    def get_study(self, study_id: str) -> Optional[dict]: ...

    @abstractmethod
    def update_study(self, study_id: str, fields: dict) -> None:
        """Update fields on a study record (e.g. config)."""
        ...

    @abstractmethod
    def delete_study(self, study_id: str) -> None:
        """Permanently delete a study and all its associated runs and structures."""
        ...

    @abstractmethod
    def list_studies(self) -> List[dict]: ...

    @abstractmethod
    def get_study_runs(self, study_id: str) -> List[dict]: ...

    # -- evaluations ----------------------------------------------------

    @abstractmethod
    def add_evaluation(
        self,
        structure_id: str,
        run_name: str,
        calculator: str,
        config_json: str,
        energy_per_atom: float,
        energy_total: float,
        min_phonon_freq: Optional[float] = None,
    ) -> str: ...

    @abstractmethod
    def clear_evaluations(
        self, run_name: str, calculator: Optional[str] = None
    ) -> None:
        """Delete evaluation records for a specific run (and optionally calculator)."""
        ...

    @abstractmethod
    def has_evaluation(
        self, structure_id: str, calculator: str, config_json: str
    ) -> bool: ...

    @abstractmethod
    def get_evaluations(
        self, run_name: str, calculator: Optional[str] = None
    ) -> List[dict]: ...

    # -- lifecycle ------------------------------------------------------

    @abstractmethod
    def close(self) -> None: ...

    # -- convenience (non-abstract) -------------------------------------
    # These delegate to the ABC primitives so that old call sites keep
    # working with any backend.  SurrealDBStore overrides most of these
    # with its own native implementations.

    def update_structure_phonon(self, struct_id: str, min_phonon_freq: float) -> None:
        self.update_structure(struct_id, min_phonon_freq=min_phonon_freq)

    def add_generation_placeholders(
        self,
        run_name: str,
        placeholders: List[tuple],
    ) -> int:
        structs = [
            {"id": cid, "status": "generating", "gen_spg": spg, "gen_fu": fu}
            for cid, spg, fu in placeholders
        ]
        return self.add_structures(run_name, structs)

    def get_pending_generation(self, run_name: str) -> List[dict]:
        rows = self.get_structures(run_name, status="generating")
        return [
            {"id": r["id"], "gen_spg": r.get("gen_spg"), "gen_fu": r.get("gen_fu")}
            for r in rows
        ]

    def get_unrelaxed_candidates(self, run_name: str) -> List[dict]:
        rows = self.get_structures(run_name, status="generated")
        return [{"id": r["id"], "atoms": r.get("atoms")} for r in rows]

    def get_run_structures(
        self,
        run_name: str,
        status: Optional[str] = None,
        symprec: float = 1e-3,
    ) -> List[dict]:
        return self.get_structures(run_name, status=status, symprec=symprec)

    def get_structures_for_analysis(
        self,
        run_id: str,
        statuses: tuple = ("relaxed",),
    ) -> List[dict]:
        """Load structures for analysis -- atoms + metadata, no vectors."""
        rows = self.get_structures(run_id, statuses=statuses)
        results: List[dict] = []
        for r in rows:
            atoms = r.get("atoms")
            if atoms is None:
                continue
            forces = atoms.info.get("initial_forces")
            if forces is not None:
                forces = np.asarray(forces)
            results.append(
                {
                    "id": r["id"],
                    "energy_per_atom": r["energy_per_atom"],
                    "atoms": atoms,
                    "forces": forces,
                }
            )
        return results

    def update_generated_structure(
        self,
        struct_id: str,
        atoms: "Atoms",
        vector: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        self.update_structure(struct_id, status="generated", atoms=atoms, vector=vector)

    def discard_generation_placeholder(self, struct_id: str) -> None:
        self.update_structure(struct_id, status="discarded")

    def get_nearby_structures(
        self,
        vector: np.ndarray,
        threshold: float,
        run_id: Optional[str] = None,
        statuses: tuple[str, ...] = ("relaxed",),
        exclude_ids: Optional[List[str]] = None,
        limit: int = 500,
        **kwargs,
    ) -> List[dict]:
        return self.find_neighbors(
            vector,
            threshold,
            k=limit,
            run_id=run_id,
            statuses=statuses,
            exclude_ids=exclude_ids,
        )

    def add_candidate(
        self,
        atoms: "Atoms",
        vector: np.ndarray,
        run_name: str,
        candidate_id: str,
        metadata: Optional[dict] = None,
        **kwargs,
    ) -> str:
        self.add_structures(
            run_name,
            [
                {
                    "id": candidate_id,
                    "status": "generated",
                    "atoms": atoms,
                    "vector": vector,
                    "metadata": metadata or {},
                }
            ],
        )
        return candidate_id

    @classmethod
    def from_path(cls, db_path: Path, **kwargs) -> "StructureStore":
        """Create a persistent store backed by a directory on disk."""
        raise NotImplementedError
