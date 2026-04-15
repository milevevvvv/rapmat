from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import numpy as np
from ase import Atoms

# ------------------------------------------------------------------ #
#  Descriptor ABC
# ------------------------------------------------------------------ #


class StructureDescriptor(ABC):
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def compute(self, atoms: Atoms) -> np.ndarray: ...

    @abstractmethod
    def code_version(self) -> str: ...

    @abstractmethod
    def descriptor_id(self) -> str: ...

    def vec_col_name(self) -> str:
        # TODO: move to storage backends
        return f"vec_{self.descriptor_id()[:12]}"


# ------------------------------------------------------------------ #
#  Store ABC
# ------------------------------------------------------------------ #


class StructureStore(ABC):
    @abstractmethod
    def register_descriptor(
        self,
        desc_id: str,
        dim: int,
        meta: Optional[dict] = None,
    ) -> str:
        # TODO: auto register on first use
        ...

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
    def update_run_config(self, name: str, config: dict) -> None: ...

    @abstractmethod
    def delete_run(self, run_name: str) -> None: ...

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

    @abstractmethod
    def add_structures(self, run_name: str, structures: List[dict]) -> int:
        # TODO: move to objects with defaults instead of just dicts
        ...

    @abstractmethod
    def update_structure(self, struct_id: str, **fields) -> None: ...

    @abstractmethod
    def clear_run_phonon_results(self, run_name: str) -> None: ...

    @abstractmethod
    def get_structures(
        self,
        run_name: str,
        *,
        status: Optional[str] = None,
        statuses: Optional[tuple[str, ...]] = None,
        fields: Optional[list[str]] = None,
        symprec: float = 1e-3,
    ) -> List[dict]: ...

    @abstractmethod
    def count(self) -> int: ...

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
    ) -> List[dict]: ...

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
    def update_study(self, study_id: str, fields: dict) -> None: ...

    @abstractmethod
    def delete_study(self, study_id: str) -> None: ...

    @abstractmethod
    def list_studies(self) -> List[dict]: ...

    @abstractmethod
    def get_study_runs(self, study_id: str) -> List[dict]: ...

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
    ) -> None: ...

    @abstractmethod
    def has_evaluation(
        self, structure_id: str, calculator: str, config_json: str
    ) -> bool: ...

    @abstractmethod
    def get_evaluations(
        self, run_name: str, calculator: Optional[str] = None
    ) -> List[dict]: ...

    @abstractmethod
    def close(self) -> None: ...

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
        raise NotImplementedError
